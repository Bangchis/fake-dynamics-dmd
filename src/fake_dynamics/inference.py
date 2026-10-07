import json
from pathlib import Path

import torch

from .backend import SDXLBackend
from .checkpoint import load_payload
from .config import Config
from .data import read_prompts, sha256_file
from .runtime import environment_manifest, write_json
from .sampling import sample_generator
from .weights import extract_strict, load_weights


@torch.no_grad()
def generate(
    config,
    checkpoint,
    prompts_path,
    output_dir,
    *,
    weights="ema",
    seed=10,
    count=None,
    initial_dmd2=False,
):
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("SDXL sample generation requires CUDA")
    if config.backend != "sdxl":
        raise ValueError("Sample command is only for SDXL")
    if weights not in ("generator", "ema"):
        raise ValueError("Choose generator or ema explicitly")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose an empty sample directory to prevent mixed checkpoint results")
    device = torch.device("cuda")
    backend = SDXLBackend(config, device, training=False)
    if initial_dmd2:
        if weights != "generator":
            raise ValueError(
                "Initial DMD2 checkpoint contains generator weights, not this run's EMA"
            )
        state = load_weights(checkpoint)
        backend.generator.unet.load_state_dict(
            extract_strict(state, backend.generator.unet.state_dict()), strict=True
        )
        checkpoint_hash = sha256_file(checkpoint)
    else:
        payload = load_payload(checkpoint)
        scientific_keys = (
            "backbone",
            "backbone_revision",
            "resolution",
            "generator_anchors",
            "prediction_type",
            "conversion_dtype",
        )
        if any(config.as_dict()[k] != payload["config"][k] for k in scientific_keys):
            raise ValueError("Sampling config differs from checkpoint scientific config")
        backend.generator.load_state_dict(payload["models"][weights], strict=True)
        if not torch.equal(backend.diffusion.alphas_cumprod.cpu(), payload["alphas_cumprod"]):
            raise ValueError("Sampling scheduler differs from checkpoint")
        checkpoint_hash = json.loads((Path(checkpoint) / "manifest.json").read_text())[
            "training_sha256"
        ]
    backend.generator.requires_grad_(False).eval()
    records = read_prompts(prompts_path)
    if count is not None:
        if not 0 < count <= len(records):
            raise ValueError("count must be positive and <= prompt count")
        records = records[:count]
    rows = []
    dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
    for index, record in enumerate(records):
        condition = backend.encode([record["prompt"]])
        generator = torch.Generator(device=device).manual_seed(seed + index)
        with torch.autocast("cuda", dtype=dtype, enabled=config.mixed_precision != "no"):
            latent = sample_generator(
                backend.generator,
                backend.diffusion,
                condition,
                (1, 4, config.resolution // 8, config.resolution // 8),
                device,
                generator,
            )
        pixels = backend.decode(latent)[0].numpy()
        filename = f"{index:06d}.png"
        Image.fromarray(pixels).save(output / filename)
        row = {
            **record,
            "file": filename,
            "seed": seed + index,
            "sha256": sha256_file(output / filename),
        }
        rows.append(row)
        # Append each completed image so a failed generation leaves a useful trace.
        with (output / "mapping.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "complete": True,
        "count": len(rows),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "weights": weights,
        "initial_dmd2": initial_dmd2,
        "prompts_sha256": sha256_file(prompts_path),
        "mapping_sha256": sha256_file(output / "mapping.jsonl"),
        "seed_base": seed,
        "seed_policy": "per_sample_seed_plus_prompt_index",
        "batch_size": 1,
        "resolution": config.resolution,
        "sampler": "dmd2_stochastic_x0_renoising_999_749_499_249",
        "external_cfg": False,
        "watermark": False,
        "quantization": "uint8_truncation_of_clamp((decoded+1)*127.5)",
        "config": config.as_dict(),
        "config_hash": config.digest(),
        "scheduler": backend.scheduler_config,
        "environment": environment_manifest(),
        "paper_protocol_equivalence": "unverified",
    }
    write_json(output / "generation_manifest.json", manifest)
    return manifest


def export_unet(checkpoint, output, weights):
    from safetensors.torch import save_file

    payload = load_payload(checkpoint)
    if weights not in ("generator", "ema"):
        raise ValueError("Export only generator or ema")
    state = payload["models"][weights]
    output = Path(output)
    if output.exists():
        raise ValueError("Export output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Strip only our wrapper prefix, keeping native Diffusers UNet key names.
    tensors = {k.removeprefix("unet."): v.detach().cpu().contiguous() for k, v in state.items()}
    save_file(
        tensors, str(output), metadata={"weights": weights, "config_hash": payload["config_hash"]}
    )
    write_json(
        output.with_suffix(".json"),
        {"weights": weights, "config": payload["config"], "sha256": sha256_file(output)},
    )


def checkpoint_config(path):
    return Config(**load_payload(path)["config"])
