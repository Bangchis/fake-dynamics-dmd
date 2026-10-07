"""Atomic directories; retain every checkpoint, including warm start and best candidates."""

import json
import os
import uuid
from pathlib import Path

import torch

from .data import sha256_file
from .runtime import restore_rng, rng_state, unwrap, write_json

SCHEMA_VERSION = 1


def save_checkpoint(trainer, label="periodic"):
    runtime = trainer.runtime
    local = {"rng": rng_state(), "data": trainer.data.state_dict(), "probe": trainer.probe_state}
    rank_states = runtime.gather(local)
    destination = (
        Path(trainer.config.output_dir)
        / "checkpoints"
        / (
            f"g{trainer.state['generator_updates']:07d}_f{trainer.state['fake_updates']:07d}_{label}"
        )
    )
    status = None
    if runtime.rank == 0:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                # Preserve an earlier checkpoint if this is a repeated boundary/save.
                destination = destination.with_name(destination.name + "_" + uuid.uuid4().hex[:8])
            temp = destination.with_name(".incomplete_" + uuid.uuid4().hex)
            temp.mkdir()
            payload = {
                "schema_version": SCHEMA_VERSION,
                "config": trainer.config.as_dict(),
                "config_hash": trainer.config.digest(),
                "world_size": runtime.world_size,
                "state": trainer.state,
                "schedule_weights": trainer.config.weights(trainer.state["generator_updates"]),
                "models": {
                    name: unwrap(getattr(trainer.backend, name)).state_dict()
                    for name in ("generator", "fake", "ema")
                },
                "optimizers": {
                    name: optimizer.state_dict() for name, optimizer in trainer.optimizers.items()
                },
                "scalers": {name: scaler.state_dict() for name, scaler in trainer.scalers.items()},
                "ranks": rank_states,
                "provenance": trainer.backend.provenance,
                "scheduler_config": trainer.backend.scheduler_config,
                "alphas_cumprod": trainer.backend.diffusion.alphas_cumprod.cpu(),
            }
            torch.save(payload, temp / "training.pt")
            write_json(
                temp / "manifest.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "config_hash": trainer.config.digest(),
                    "state": trainer.state,
                    "schedule_weights": payload["schedule_weights"],
                    "training_sha256": sha256_file(temp / "training.pt"),
                    "provenance": trainer.backend.provenance,
                },
            )
            os.replace(temp, destination)
            write_json(
                Path(trainer.config.output_dir) / "latest.json", {"checkpoint": str(destination)}
            )
            trainer.log("checkpoint_saved", path=str(destination))
            status = {"ok": True, "path": str(destination)}
        except Exception as error:
            # Tell all live ranks rather than leaving them waiting at a success barrier.
            # An OS/process kill still requires the recipient's distributed launcher cleanup.
            status = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    status = runtime.broadcast(status)
    if not status["ok"]:
        raise RuntimeError(f"Checkpoint write failed on rank 0: {status['error']}")
    return Path(status["path"])


def load_payload(path, *, verify=True):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if verify and sha256_file(path / "training.pt") != manifest["training_sha256"]:
        raise ValueError("Checkpoint checksum mismatch; use a complete checkpoint directory")
    payload = torch.load(path / "training.pt", map_location="cpu", weights_only=True, mmap=True)
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema")
    return payload


def resume(trainer, path):
    payload = load_payload(path)
    if payload["world_size"] != trainer.runtime.world_size:
        raise ValueError("Exact resume requires the same world size")
    # Runtime budget/output/log intervals may change; scientific and data config must match.
    allowed = {
        "output_dir",
        "total_generator_updates",
        "checkpoint_interval_g_updates",
        "probe_interval_g_updates",
        "hardware_and_compute_budget",
    }
    current = trainer.config.as_dict()
    changes = [
        key for key in current if key not in allowed and current[key] != payload["config"][key]
    ]
    if changes:
        raise ValueError(
            f"Resume config mismatch: {changes}. Start a new warm-start experiment instead."
        )
    coefficients = trainer.backend.diffusion.alphas_cumprod.cpu()
    if not torch.equal(coefficients, payload["alphas_cumprod"]):
        raise ValueError("Resume scheduler coefficients differ")
    for name, state in payload["models"].items():
        unwrap(getattr(trainer.backend, name)).load_state_dict(state, strict=True)
    for name, state in payload["optimizers"].items():
        trainer.optimizers[name].load_state_dict(state)
    for name, state in payload["scalers"].items():
        trainer.scalers[name].load_state_dict(state)
    trainer.state = payload["state"]
    rank_state = payload["ranks"][trainer.runtime.rank]
    trainer.data.load_state_dict(rank_state["data"])
    trainer.probe_state = rank_state["probe"]
    trainer.backend.provenance = payload["provenance"]
    trainer.backend.teacher.requires_grad_(False).eval()
    trainer.backend.ema.requires_grad_(False).eval()
    trainer.backend.generator.train()
    trainer.backend.fake.train()
    restore_rng(rank_state["rng"])
    trainer.log("resumed", path=str(path), source_config_hash=payload["config_hash"])
