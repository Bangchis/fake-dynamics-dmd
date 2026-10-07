"""Strict paired checkpoint import; discriminator/dummy/teacher keys never become F."""

from collections import Counter
from pathlib import Path

import torch

from .data import sha256_file


def load_weights(path):
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        # Restricted unpickler. Do not silently retry with weights_only=False.
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
        raise ValueError("Expected a tensor state dict, optionally wrapped in state_dict")
    if state and all(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    return state


def extract_strict(state, expected, prefix=""):
    selected = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    missing = sorted(set(expected) - set(selected))
    extra = sorted(set(selected) - set(expected))
    bad_shapes = [
        k
        for k in set(expected) & set(selected)
        if not isinstance(selected[k], torch.Tensor) or selected[k].shape != expected[k].shape
    ]
    if missing or extra or bad_shapes:
        raise ValueError(
            f"Checkpoint mismatch for prefix {prefix!r}: missing={missing[:12]}, "
            f"extra={extra[:12]}, wrong_shape={bad_shapes[:12]}"
        )
    return selected


def import_paired(generator, fake, config):
    source_g = load_weights(config.generator_checkpoint)
    g_state = extract_strict(source_g, generator.state_dict())
    generator.load_state_dict(g_state, strict=True)
    provenance = {
        "generator": {
            "path": str(config.generator_checkpoint),
            "sha256": sha256_file(config.generator_checkpoint),
        }
    }
    del g_state, source_g
    if config.fake_initialization == "paired":
        source_f = load_weights(config.guidance_checkpoint)
        f_state = extract_strict(source_f, fake.state_dict(), "fake_unet.")
        fake.load_state_dict(f_state, strict=True)
        provenance["fake"] = {
            "path": str(config.guidance_checkpoint),
            "sha256": sha256_file(config.guidance_checkpoint),
            "prefix": "fake_unet.",
            "discarded_key_count": len(source_f) - len(f_state),
        }
    else:
        from diffusers import UNet2DConditionModel

        teacher = UNet2DConditionModel.from_pretrained(
            config.backbone, subfolder="unet", revision=config.backbone_revision
        )
        fake.load_state_dict(teacher.state_dict(), strict=True)
        provenance["fake"] = {
            "kind": "teacher_fallback",
            "warmup_updates": config.fake_warmup_updates,
        }
    return provenance


def inspect_weights(path):
    state = load_weights(path)
    return {
        "path": str(path),
        "tensor_count": len(state),
        "prefix_counts": dict(Counter(k.split(".")[0] for k in state)),
        "first_tensors": {
            k: {"shape": list(v.shape), "dtype": str(v.dtype)}
            for k, v in list(state.items())[:12]
            if isinstance(v, torch.Tensor)
        },
    }
