"""Recipient-only numerical checks. Never executed on the author's Mac."""

import json
from dataclasses import replace

import pytest
import torch

import fake_dynamics.trainer as training
from fake_dynamics.config import Config
from fake_dynamics.ema import ema_master
from fake_dynamics.runtime import Runtime, unwrap
from fake_dynamics.trainer import Trainer


def config(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(json.dumps({"prompt": f"p{i}"}) + "\n" for i in range(16)))
    return Config(
        backend="toy",
        resolution=32,
        mixed_precision="no",
        prompts_path=str(path),
        output_dir=str(tmp_path / "run"),
        per_device_batch_size=2,
        gradient_accumulation_steps=3,
        target_global_batch_size=6,
        total_generator_updates=2,
        checkpoint_interval_g_updates=1,
        max_grad_norm=2.0,
    )


def test_mean_gradient_clipped_once_and_ema_updated_once(tmp_path, monkeypatch):
    trainer = Trainer(config(tmp_path))
    parameter = next(trainer.backend.generator.parameters())
    reference = torch.nn.Parameter(parameter.detach().clone())
    reference_optimizer = torch.optim.AdamW(
        [reference],
        lr=trainer.config.generator_lr,
        betas=trainer.config.adam_betas,
        weight_decay=trainer.config.weight_decay,
        foreach=False,
    )
    # A full-batch mean, deliberately above the clipping threshold. If clipping
    # happens per microbatch or loss/count is omitted, the pre-clip norm differs.
    (reference.flatten()[0] * 4).backward()
    reference_norm = torch.nn.utils.clip_grad_norm_([reference], 2.0)
    reference_optimizer.step()
    coefficients = iter((1.0, 3.0, 8.0))
    ema_calls = []
    original_update = training.update_ema

    def update(*args):
        ema_calls.append(1)
        return original_update(*args)

    monkeypatch.setattr(training, "update_ema", update)

    def microbatch():
        loss = parameter.flatten()[0] * next(coefficients)
        return loss, {"loss": float(loss.detach()), "anchor": 999}

    success, metrics = trainer.accumulated_step("generator", microbatch)
    assert success and metrics["grad_norm"] == pytest.approx(float(reference_norm))
    torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
    assert ema_calls == [1]
    assert trainer.state["generator_updates"] == trainer.state["generator_attempts"] == 1
    assert trainer.state["fake_updates"] == 0
    assert metrics["local_samples_processed"] == metrics["global_batch_size"] == 6
    assert all(p.grad is None for p in trainer.backend.generator.parameters())
    trainer.runtime.close()


def test_nonfinite_discards_entire_window_and_retry_has_no_stale_gradients(tmp_path):
    trainer = Trainer(config(tmp_path))
    parameter = next(trainer.backend.generator.parameters())
    before = {k: v.clone() for k, v in unwrap(trainer.backend.generator).state_dict().items()}
    ema_before = {k: v.clone() for k, v in ema_master(trainer.backend.ema).state_dict().items()}
    coefficients = iter((1.0, float("nan"), 8.0))

    def microbatch():
        loss = parameter.flatten()[0] * next(coefficients)
        return loss, {"loss": float(loss.detach()), "anchor": 999}

    success, metrics = trainer.accumulated_step("generator", microbatch)
    assert not success and metrics["microbatches_processed"] == 2
    assert trainer.state["generator_updates"] == 0
    assert trainer.state["generator_attempts"] == 1
    assert not trainer.optimizers["generator"].state
    for key, value in unwrap(trainer.backend.generator).state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    for key, value in ema_master(trainer.backend.ema).state_dict().items():
        torch.testing.assert_close(value, ema_before[key], rtol=0, atol=0)
    assert all(p.grad is None for p in trainer.backend.generator.parameters())
    coefficients = iter((1.0, 1.0, 1.0))
    success, metrics = trainer.accumulated_step("generator", microbatch)
    assert success and metrics["grad_norm"] == pytest.approx(1.0)
    assert trainer.state["generator_updates"] == 1
    trainer.runtime.close()


def test_real_windows_keep_anchor_weights_and_targets_fixed(tmp_path, monkeypatch):
    trainer = Trainer(config(tmp_path))
    observed = []
    original = trainer.fresh_input

    def fresh():
        condition, anchor, h = original()
        observed.append((anchor, trainer.state["generator_updates"], trainer.state["fake_updates"]))
        return condition, anchor, h

    monkeypatch.setattr(trainer, "fresh_input", fresh)
    assert trainer.fake_step(0.05)
    assert len({row[0] for row in observed}) == 1
    assert all(row[1:] == (0, 0) for row in observed)
    observed.clear()
    assert trainer.generator_step(0.05, 0.1)
    assert len({row[0] for row in observed}) == 1
    assert all(row[1:] == (0, 1) for row in observed)
    rows = [json.loads(line) for line in trainer.logger.path.read_text().splitlines()]
    assert all(row["microbatches_processed"] == 3 for row in rows)
    assert rows[-1]["beta"] == 0.05 and rows[-1]["lambda_cd"] == 0.1
    trainer.runtime.close()


def test_target_batch_mismatch_fails_before_allocating_models(tmp_path):
    cfg = replace(config(tmp_path), target_global_batch_size=128)
    with pytest.raises(ValueError, match="Effective global batch"):
        Runtime(cfg)


def test_sparse_error_reduction_is_weighted(tmp_path):
    trainer = Trainer(config(tmp_path))
    result = trainer.reduce_microbatch_metrics(
        [
            {"anchor": 999, "loss": 1, "corrected_error_0_249": 2, "samples_0_249": 1},
            {"anchor": 999, "loss": 3, "corrected_error_0_249": 8, "samples_0_249": 3},
            {"anchor": 999, "loss": 5, "corrected_error_0_249": None, "samples_0_249": 0},
        ],
        3,
    )
    assert result["corrected_error_0_249"] == 6.5 and result["samples_0_249"] == 4
    assert result["loss"] == 3
    trainer.runtime.close()
