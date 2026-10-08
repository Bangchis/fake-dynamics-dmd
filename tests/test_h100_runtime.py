"""Authored for recipient execution. No local Mac execution was performed."""

import json
import os
import socket
from dataclasses import replace

import pytest
import torch

from fake_dynamics.backend import ToyEpsilon
from fake_dynamics.checkpoint import load_payload, save_checkpoint
from fake_dynamics.config import Config
from fake_dynamics.ema import OffloadedEMA, ema_master
from fake_dynamics.runtime import unwrap
from fake_dynamics.trainer import Trainer, update_ema


def cfg(root, name):
    return Config(
        backend="toy",
        resolution=32,
        prompts_path=str(root / "prompts.jsonl"),
        output_dir=str(root / name),
        per_device_batch_size=2,
        total_generator_updates=2,
        checkpoint_interval_g_updates=1,
        mixed_precision="no",
        ema_device="cpu",
        ema_forward_dtype="float32",
        beta_override=0.05,
        consistency_override=0.1,
    )


def prompts(root):
    (root / "prompts.jsonl").write_text(
        "".join(json.dumps({"id": str(i), "prompt": f"caption {i}"}) + "\n" for i in range(16))
    )


def cycle(trainer):
    beta, cd = trainer.config.weights(trainer.state["generator_updates"])
    while trainer.state["fake_in_cycle"] < 5:
        assert trainer.fake_step(beta)
        trainer.state["fake_in_cycle"] += 1
    assert trainer.generator_step(beta, cd)
    trainer.state["fake_in_cycle"] = 0


def test_cpu_ema_snapshot_invalidation_and_full_precision(tmp_path):
    prompts(tmp_path)
    trainer = Trainer(cfg(tmp_path, "ema"))
    ema = trainer.backend.ema
    condition = trainer.backend.encode(["a", "b"])
    x, times = torch.randn(trainer.shape), torch.tensor([999, 749])
    reference = ToyEpsilon()
    reference.load_state_dict(ema.master.state_dict())
    torch.testing.assert_close(ema(x, times, condition), reference(x, times, condition))
    before = {k: v.clone() for k, v in ema.master.state_dict().items()}
    assert ema._snapshot is not None
    with torch.no_grad():
        for parameter in trainer.backend.generator.parameters():
            parameter.add_(0.01)
    update_ema(ema, trainer.backend.generator, 0.99)
    assert ema._snapshot is None
    for key, value in ema.master.state_dict().items():
        assert value.device.type == "cpu" and value.dtype == torch.float32
        torch.testing.assert_close(
            value, before[key] * 0.99 + trainer.backend.generator.state_dict()[key] * 0.01
        )
    trainer.tensorboard.close()
    trainer.runtime.close()


def test_pruning_is_opt_in_and_protects_pinned_and_failure_checkpoints(tmp_path):
    prompts(tmp_path)
    trainer = Trainer(replace(cfg(tmp_path, "retention"), checkpoint_keep_last=1))
    pinned = save_checkpoint(trainer)
    (pinned / ".pin").touch()
    failure = save_checkpoint(trainer, "failed")
    previous = save_checkpoint(trainer)
    final = save_checkpoint(trainer, "final")
    assert pinned.exists() and failure.exists() and final.exists()
    assert not previous.exists()
    assert json.loads((tmp_path / "retention/latest.json").read_text())["checkpoint"] == str(final)
    assert (
        load_payload(final)["models"]["ema"].keys()
        == ema_master(trainer.backend.ema).state_dict().keys()
    )
    trainer.runtime.close()


def distributed_worker(rank, root, port):
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    config = replace(
        cfg(root, "first"),
        distributed_strategy="ddp",
        optimizer_state_sharding="zero1",
        gradient_accumulation_steps=2,
        target_global_batch_size=8,
    )
    # Compare DDP/no_sync/zero1 gradients against an independently differentiated
    # full effective batch. Exact resume alone cannot catch a wrong gradient scale.
    mean = Trainer(replace(config, output_dir=str(root / "mean")))
    reference = ToyEpsilon()
    reference.load_state_dict(unwrap(mean.backend.fake).state_dict())
    x = torch.ones(mean.shape)
    times = torch.full((mean.shape[0],), 999, dtype=torch.long)
    condition = mean.backend.encode(["a", "b"])
    coefficients = iter((1 + 4 * rank, 3 + 4 * rank))

    def microbatch():
        coefficient = next(coefficients)
        expected_loss = reference(x, times, condition).square().mean() * coefficient / 2
        expected_loss.backward()
        loss = mean.backend.fake(x, times, condition).square().mean() * coefficient
        return loss, {"loss": float(loss.detach()), "anchor": 999}

    original_step = mean.optimizer_step

    def checked_step(name, loss, **kwargs):
        expected_grad = torch.cat([p.grad.flatten() for p in reference.parameters()])
        torch.distributed.all_reduce(expected_grad)
        expected_grad /= 2
        actual_grad = torch.cat([p.grad.flatten() for p in mean.backend.fake.parameters()])
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)
        return original_step(name, loss, **kwargs)

    mean.optimizer_step = checked_step
    assert mean.accumulated_step("fake", microbatch)[0]
    mean.optimizer_step = original_step
    # Only rank 0 is invalid. Exercise both an early no_sync rejection and a
    # final synchronized rejection; a valid retry must not inherit partial grads
    # or leave the DDP reducer waiting for hooks from the discarded window.
    for failure_index in (0, 1):
        before = {k: v.clone() for k, v in unwrap(mean.backend.fake).state_dict().items()}
        updates_before = mean.state["fake_updates"]
        coefficients = iter(
            float("nan") if rank == 0 and index == failure_index else 1.0 for index in range(2)
        )

        def invalid_microbatch():
            loss = mean.backend.fake(x, times, condition).square().mean() * next(coefficients)
            return loss, {"loss": float(loss.detach()), "anchor": 999}

        assert not mean.accumulated_step("fake", invalid_microbatch)[0]
        assert mean.state["fake_updates"] == updates_before
        for key, value in unwrap(mean.backend.fake).state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        assert all(p.grad is None for p in mean.backend.fake.parameters())
        coefficients = iter((1.0, 1.0))
        assert mean.accumulated_step("fake", invalid_microbatch)[0]
        assert mean.state["fake_updates"] == updates_before + 1
    mean.runtime.close()
    first = Trainer(config)
    combined = first.runtime.aggregate_metrics(
        {
            "k_F": 0,
            "k_G": 0,
            "rank": rank,
            "corrected_error_0_249": 2 if rank == 0 else 8,
            "samples_0_249": 1 if rank == 0 else 3,
            "cuda_peak_allocated_gib": rank + 1,
            "forward_counts": {"fake": rank + 2},
        }
    )
    assert combined["corrected_error_0_249"] == 6.5
    assert combined["samples_0_249"] == 4
    assert combined["cuda_peak_allocated_gib"] == 2
    assert combined["forward_counts/fake"] == 5
    cycle(first)
    checkpoint = save_checkpoint(first, "test")
    cycle(first)
    expected = {
        name: {
            k: v.clone()
            for k, v in (
                ema_master(getattr(first.backend, name))
                if name == "ema"
                else unwrap(getattr(first.backend, name))
            )
            .state_dict()
            .items()
        }
        for name in ("generator", "fake", "ema")
    }
    expected_rng = torch.randn(8)
    expected_prompt = first.data.next()
    first.runtime.close()
    restored = Trainer(replace(config, output_dir=str(root / "restored")), checkpoint)
    cycle(restored)
    assert restored.state["fake_updates"] == 10 and restored.state["generator_updates"] == 2
    for name, state in expected.items():
        model = (
            ema_master(getattr(restored.backend, name))
            if name == "ema"
            else unwrap(getattr(restored.backend, name))
        )
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    assert torch.equal(torch.randn(8), expected_rng)
    assert restored.data.next() == expected_prompt
    restored.runtime.close()


def test_two_rank_zero1_resume_replays_models_rng_and_prompts(tmp_path):
    prompts(tmp_path)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    torch.multiprocessing.spawn(distributed_worker, args=(tmp_path, port), nprocs=2, join=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ema_device", "disk"),
        ("teacher_weight_dtype", "float16"),
        ("checkpoint_keep_last", 0),
        ("optimizer_state_sharding", "zero3"),
    ],
)
def test_invalid_runtime_settings_fail(field, value):
    config = Config()
    setattr(config, field, value)
    with pytest.raises(ValueError):
        config.validate()


def test_cpu_ema_wrapper_is_frozen():
    ema = OffloadedEMA(ToyEpsilon(), torch.device("cpu"), torch.float32)
    assert not ema.training and all(not p.requires_grad for p in ema.parameters())
