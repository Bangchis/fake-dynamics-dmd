"""Atomic rank-local optimizer checkpoints, opt-in bounded retention, v1 model reading."""

import json
import os
import shutil
import time
import uuid
from pathlib import Path

import torch

from .data import sha256_file
from .ema import ema_master, release_ema
from .runtime import environment_manifest, restore_rng, rng_state, unwrap, write_json

SCHEMA_VERSION = 2


def state_model(backend, name):
    model = unwrap(getattr(backend, name))
    return ema_master(model) if name == "ema" else model


def prune_checkpoints(trainer, latest):
    keep = trainer.config.checkpoint_keep_last
    if keep is None:
        return
    root = Path(trainer.config.output_dir) / "checkpoints"
    candidates = []
    for folder in root.iterdir():
        if (
            folder.name.startswith(".")
            or folder.is_symlink()
            or not folder.is_dir()
            or (folder / ".pin").exists()
        ):
            continue
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            metadata = json.loads(manifest_path.read_text())
        except (ValueError, OSError):
            continue  # A malformed/unrecognized folder is never a deletion candidate.
        if not isinstance(metadata, dict) or not isinstance(metadata.get("saved_unix_ns"), int):
            continue
        if metadata.get("managed_by") != "fdmd-v2" or metadata.get("label") not in (
            "periodic",
            "final",
        ):
            continue
        candidates.append((metadata["saved_unix_ns"], folder))
    candidates.sort(key=lambda item: item[0], reverse=True)
    for _, folder in candidates[keep:]:
        if folder.resolve() == latest.resolve() or folder.resolve().parent != root.resolve():
            continue
        shutil.rmtree(folder)
        trainer.log("checkpoint_pruned", path=str(folder), policy_keep_last=keep)


def save_checkpoint(trainer, label="periodic"):
    started = time.monotonic()
    runtime = trainer.runtime
    release_ema(trainer.backend.ema)
    preparation = None
    if runtime.rank == 0:
        try:
            root = Path(trainer.config.output_dir) / "checkpoints"
            root.mkdir(parents=True, exist_ok=True)
            name = f"g{trainer.state['generator_updates']:07d}_f{trainer.state['fake_updates']:07d}_{label}"
            destination = root / name
            if destination.exists():
                destination = root / (name + "_" + uuid.uuid4().hex[:8])
            temp = root / (".incomplete_" + uuid.uuid4().hex)
            temp.mkdir()
            preparation = {"ok": True, "temp": str(temp), "destination": str(destination)}
        except Exception as error:
            preparation = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    preparation = runtime.broadcast(preparation)
    if not preparation["ok"]:
        raise RuntimeError(f"Checkpoint preparation failed: {preparation['error']}")
    temp, destination = Path(preparation["temp"]), Path(preparation["destination"])
    local_status = None
    try:
        local = {
            "rank": runtime.rank,
            "torch_version": str(torch.__version__),
            "optimizers": {
                name: runtime.optimizer_checkpoint_state(opt, getattr(trainer.backend, name))
                for name, opt in trainer.optimizers.items()
            },
            "scalers": {name: scaler.state_dict() for name, scaler in trainer.scalers.items()},
            "rng": rng_state(),
            "data": trainer.data.state_dict(),
            "probe": trainer.probe_state,
            "forward_counts": dict(trainer.state["forward_counts"]),
        }
        filename = f"rank{runtime.rank:03d}.pt"
        torch.save(local, temp / filename)
        local_status = {
            "ok": True,
            "file": filename,
            "sha256": sha256_file(temp / filename),
            "bytes": (temp / filename).stat().st_size,
        }
        if runtime.rank == 0:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "config": trainer.config.as_dict(),
                "config_hash": trainer.config.digest(),
                "world_size": runtime.world_size,
                "state": trainer.state,
                "schedule_weights": trainer.config.weights(trainer.state["generator_updates"]),
                "models": {
                    name: state_model(trainer.backend, name).state_dict()
                    for name in ("generator", "fake", "ema")
                },
                "provenance": trainer.backend.provenance,
                "scheduler_config": trainer.backend.scheduler_config,
                "alphas_cumprod": trainer.backend.diffusion.alphas_cumprod.cpu(),
            }
            torch.save(payload, temp / "training.pt")
    except Exception as error:
        local_status = {
            "ok": False,
            "rank": runtime.rank,
            "error": f"{type(error).__name__}: {error}",
        }
    rank_statuses = runtime.gather(local_status)
    status = None
    if runtime.rank == 0:
        try:
            failures = [row for row in rank_statuses if not row["ok"]]
            if failures:
                raise RuntimeError(str(failures))
            write_json(
                temp / "manifest.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "managed_by": "fdmd-v2",
                    "label": label,
                    "saved_unix_ns": time.time_ns(),
                    "config_hash": trainer.config.digest(),
                    "config": trainer.config.as_dict(),
                    "world_size": runtime.world_size,
                    "global_batch_size": runtime.effective_batch_size,
                    "prompt_sha256": trainer.data.file_hash,
                    "scheduler": trainer.backend.scheduler_config,
                    "environment": environment_manifest(),
                    "state": trainer.state,
                    "schedule_weights": payload["schedule_weights"],
                    "training_sha256": sha256_file(temp / "training.pt"),
                    "rank_files": rank_statuses,
                    "provenance": trainer.backend.provenance,
                },
            )
            os.replace(temp, destination)
            write_json(
                Path(trainer.config.output_dir) / "latest.json", {"checkpoint": str(destination)}
            )
            trainer.log(
                "checkpoint_saved",
                path=str(destination),
                label=label,
                write_seconds=time.monotonic() - started,
                total_bytes=(destination / "training.pt").stat().st_size
                + sum(row["bytes"] for row in rank_statuses),
            )
            prune_checkpoints(trainer, destination)
            status = {"ok": True, "path": str(destination)}
        except Exception as error:
            status = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    status = runtime.broadcast(status)
    if not status["ok"]:
        raise RuntimeError(f"Checkpoint write failed: {status['error']}")
    return Path(status["path"])


def load_payload(path, *, verify=True):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if verify and sha256_file(path / "training.pt") != manifest["training_sha256"]:
        raise ValueError("Checkpoint checksum mismatch; use a complete checkpoint directory")
    payload = torch.load(path / "training.pt", map_location="cpu", weights_only=True, mmap=True)
    if payload["schema_version"] not in (1, SCHEMA_VERSION):
        raise ValueError("Unsupported checkpoint schema")
    return payload


def resume(trainer, path):
    from .config import Config

    payload = load_payload(path)
    if payload["world_size"] != trainer.runtime.world_size:
        raise ValueError("Exact resume requires the same world size")
    allowed = {
        "output_dir",
        "total_generator_updates",
        "checkpoint_interval_g_updates",
        "probe_interval_g_updates",
        "hardware_and_compute_budget",
        "tensorboard",
        "tensorboard_flush_seconds",
        "sample_interval_g_updates",
        "fixed_sample_count",
        "performance_interval_g_updates",
        "checkpoint_keep_last",
    }
    source_config = Config(**payload["config"]).as_dict()
    current = trainer.config.as_dict()
    changes = [key for key in current if key not in allowed and current[key] != source_config[key]]
    if changes:
        raise ValueError(
            f"Resume config mismatch: {changes}. Start a new warm-start experiment instead."
        )
    if not torch.equal(trainer.backend.diffusion.alphas_cumprod.cpu(), payload["alphas_cumprod"]):
        raise ValueError("Resume scheduler coefficients differ")
    for name, state in payload["models"].items():
        state_model(trainer.backend, name).load_state_dict(state, strict=True)
    if payload["schema_version"] == 1:
        if trainer.config.optimizer_state_sharding != "none":
            raise ValueError("V1 full optimizer cannot resume into rank-local zero1")
        rank_state = payload["ranks"][trainer.runtime.rank]
        for name, state in payload["optimizers"].items():
            trainer.optimizers[name].load_state_dict(state)
        scaler_states = payload["scalers"]
    else:
        manifest = json.loads((Path(path) / "manifest.json").read_text())
        filename = f"rank{trainer.runtime.rank:03d}.pt"
        info = next(row for row in manifest["rank_files"] if row["file"] == filename)
        if sha256_file(Path(path) / filename) != info["sha256"]:
            raise ValueError("Rank-local optimizer/RNG checksum mismatch")
        rank_state = torch.load(
            Path(path) / filename, map_location="cpu", weights_only=True, mmap=True
        )
        if rank_state["rank"] != trainer.runtime.rank or rank_state["torch_version"] != str(
            torch.__version__
        ):
            raise ValueError("Exact rank-local resume requires identical rank and Torch version")
        for name, saved in rank_state["optimizers"].items():
            trainer.runtime.load_optimizer_checkpoint_state(
                trainer.optimizers[name], getattr(trainer.backend, name), saved
            )
        scaler_states = rank_state["scalers"]
    for name, state in scaler_states.items():
        trainer.scalers[name].load_state_dict(state)
    trainer.state = payload["state"]
    trainer.state["forward_counts"] = rank_state.get(
        "forward_counts", trainer.state["forward_counts"]
    )
    trainer.data.load_state_dict(rank_state["data"])
    trainer.probe_state = rank_state["probe"]
    trainer.backend.provenance = payload["provenance"]
    trainer.backend.teacher.requires_grad_(False).eval()
    trainer.backend.ema.requires_grad_(False).eval()
    release_ema(trainer.backend.ema)
    trainer.backend.generator.train()
    trainer.backend.fake.train()
    restore_rng(rank_state["rng"])
    trainer.log("resumed", path=str(path), source_config_hash=payload["config_hash"])
