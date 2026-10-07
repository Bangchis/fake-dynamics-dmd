import json

import pytest
import torch

from fake_dynamics.checkpoint import load_payload, resume, save_checkpoint
from fake_dynamics.config import Config
from fake_dynamics.data import PromptStream
from fake_dynamics.runtime import unwrap
from fake_dynamics.trainer import Trainer, update_ema
from fake_dynamics.weights import extract_strict


def make_config(tmp_path, name="run"):
    prompts = tmp_path / "prompts.jsonl"
    if not prompts.exists():
        prompts.write_text(
            "".join(json.dumps({"prompt": f"caption {i}", "id": str(i)}) + "\n" for i in range(12))
        )
    return Config(
        backend="toy",
        resolution=32,
        prompts_path=str(prompts),
        output_dir=str(tmp_path / name),
        per_device_batch_size=2,
        total_generator_updates=2,
        checkpoint_interval_g_updates=1,
        mixed_precision="no",
        beta_override=0.05,
        consistency_override=0.1,
    )


def cycle(trainer):
    beta, cd = trainer.config.weights(trainer.state["generator_updates"])
    while trainer.state["fake_in_cycle"] < trainer.config.fake_updates_per_generator:
        assert trainer.fake_step(beta)
        trainer.state["fake_in_cycle"] += 1
    assert trainer.generator_step(beta, cd)
    trainer.state["fake_in_cycle"] = 0


def test_gradient_ownership_for_both_losses(tmp_path, monkeypatch):
    trainer = Trainer(make_config(tmp_path))
    observed = []

    def capture(name, loss):
        loss.backward()
        for model_name in ("generator", "fake", "teacher", "ema"):
            grads = [p.grad for p in getattr(trainer.backend, model_name).parameters()]
            if model_name == name:
                assert any(g is not None and torch.isfinite(g).all() for g in grads)
            else:
                assert all(g is None for g in grads)
        observed.append(name)
        trainer.optimizers[name].zero_grad(set_to_none=True)
        return True, 1.0

    monkeypatch.setattr(trainer, "optimizer_step", capture)
    trainer.fake_step(0.05)
    trainer.generator_step(0.05, 0.1)
    assert observed == ["fake", "generator"]
    assert not trainer.backend.teacher.training
    assert not trainer.backend.ema.training


def test_nonfinite_step_keeps_counter_and_ema(tmp_path):
    trainer = Trainer(make_config(tmp_path))
    before = {k: v.clone() for k, v in trainer.backend.ema.state_dict().items()}
    parameter = next(trainer.backend.generator.parameters())
    success, _ = trainer.optimizer_step("generator", parameter.sum() * float("nan"))
    assert not success
    assert trainer.state["generator_updates"] == 0
    assert trainer.state["generator_attempts"] == 1
    for key, value in trainer.backend.ema.state_dict().items():
        assert torch.equal(value, before[key])


def test_ema_is_fp32_and_only_generator_step_updates_it(tmp_path):
    trainer = Trainer(make_config(tmp_path))
    before = {k: v.clone() for k, v in trainer.backend.ema.state_dict().items()}
    assert trainer.fake_step(0.05)
    assert all(torch.equal(v, before[k]) for k, v in trainer.backend.ema.state_dict().items())
    with torch.no_grad():
        for p in trainer.backend.generator.parameters():
            p.add_(1)
    update_ema(trainer.backend.ema, trainer.backend.generator, 0.99)
    assert all(
        p.dtype == torch.float32 and not p.requires_grad for p in trainer.backend.ema.parameters()
    )
    for key, value in trainer.backend.ema.state_dict().items():
        expected = before[key] * 0.99 + unwrap(trainer.backend.generator).state_dict()[key] * 0.01
        torch.testing.assert_close(value, expected)


def test_checkpoint_resume_replays_next_cycle_and_rng(tmp_path):
    first = Trainer(make_config(tmp_path, "first"))
    cycle(first)
    checkpoint = save_checkpoint(first, "test")
    cycle(first)
    expected = {
        name: {k: v.clone() for k, v in getattr(first.backend, name).state_dict().items()}
        for name in ("generator", "fake", "ema")
    }
    expected_rng = torch.randn(8)
    second = Trainer(make_config(tmp_path, "second"), resume_path=checkpoint)
    assert second.state["generator_updates"] == 1
    cycle(second)
    assert second.state["generator_updates"] == 2
    assert second.state["fake_updates"] == 10
    for name, state in expected.items():
        for key, value in getattr(second.backend, name).state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    assert torch.equal(torch.randn(8), expected_rng)


def test_resume_restores_mid_cycle_slots(tmp_path):
    trainer = Trainer(make_config(tmp_path))
    for _ in range(3):
        assert trainer.fake_step(0.05)
        trainer.state["fake_in_cycle"] += 1
    path = save_checkpoint(trainer, "midcycle")
    restored = Trainer(make_config(tmp_path, "restored"), resume_path=path)
    assert restored.state["fake_in_cycle"] == 3
    cycle(restored)
    assert restored.state["fake_updates"] == 5
    assert restored.state["generator_updates"] == 1


def test_cd_skips_anchor_249(tmp_path, monkeypatch):
    trainer = Trainer(make_config(tmp_path))
    condition = trainer.backend.encode(["one", "two"])
    monkeypatch.setattr(
        trainer, "fresh_input", lambda: (condition, 249, torch.randn(trainer.shape))
    )
    assert trainer.generator_step(0.05, 0.1)
    assert trainer.state["forward_counts"]["ema"] == 0
    rows = [json.loads(line) for line in trainer.logger.path.read_text().splitlines()]
    assert rows[-1]["loss_cd"] == 0 and not rows[-1]["cd_active"]


def test_prompt_only_stream_resume_and_disjoint_ranks(tmp_path):
    cfg = make_config(tmp_path)
    rank0 = PromptStream(cfg.prompts_path, 2, rank=0, world_size=2)
    rank1 = PromptStream(cfg.prompts_path, 2, rank=1, world_size=2)
    assert set(rank0.next()).isdisjoint(rank1.next())
    state = rank0.state_dict()
    expected = rank0.next()
    rank0.load_state_dict(state)
    assert rank0.next() == expected
    assert "lmdb" not in rank0.__dict__ and "images" not in rank0.__dict__


def test_import_selects_fake_only_and_shape_mismatch_fails():
    expected = {"weight": torch.zeros(2, 3)}
    state = {
        "fake_unet.weight": torch.ones(2, 3),
        "cls_pred_branch.weight": torch.ones(5),
        "real_unet.weight": torch.ones(2, 3),
    }
    selected = extract_strict(state, expected, "fake_unet.")
    assert set(selected) == {"weight"}
    state["fake_unet.weight"] = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="wrong_shape"):
        extract_strict(state, expected, "fake_unet.")


def test_resume_rejects_scientific_config_change(tmp_path):
    trainer = Trainer(make_config(tmp_path))
    path = save_checkpoint(trainer, "test")
    trainer.config.teacher_cfg = 6
    with pytest.raises(ValueError, match="teacher_cfg"):
        resume(trainer, path)
    assert load_payload(path)["state"]["generator_updates"] == 0
