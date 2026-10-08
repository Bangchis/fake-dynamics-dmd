"""Merge a pilot overlay into recipient config; never imports Torch or starts training."""

import argparse
from pathlib import Path

import yaml

MODES = {
    "b32-noaccum": (8, 1, 32),
    "b32-accum2": (4, 2, 32),
    "b16-noaccum": (4, 1, 16),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, required=True, help="Recipient YAML or run_manifest.json"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New local YAML; never overwritten"
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Unused output directory for a fresh run"
    )
    parser.add_argument("--mode", choices=MODES, default="b32-noaccum")
    parser.add_argument("--activation-checkpointing", choices=("on", "off"), default="on")
    args = parser.parse_args()

    payload = yaml.safe_load(args.base.read_text())
    if not isinstance(payload, dict):
        parser.error("Base must contain a config mapping")
    cfg = payload.get("config", payload)
    if not isinstance(cfg, dict) or cfg.get("backend") != "sdxl":
        parser.error("Base must be the recipient's resolved SDXL config")
    if cfg.get("train_weight_dtype") not in ("bfloat16", "bf16"):
        parser.error(
            "Requires the recipient's BF16-online/FP32-master backend and config; "
            "public FP32 templates alone do not implement that backend"
        )
    for key in ("generator_checkpoint", "guidance_checkpoint", "prompts_path"):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            parser.error(f"Resolved base must provide {key}")
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output config already exists; choose a new filename")
    if args.run_dir.exists() or args.run_dir.is_symlink():
        parser.error("Run directory already exists; choose a new directory")
    if args.output.resolve().is_relative_to(args.run_dir.resolve()):
        parser.error("Save the config outside the fresh run directory")

    overlay_path = (
        Path(__file__).resolve().parents[1] / "configs" / "recipient_fast_pilot_overrides.yaml"
    )
    overlay = yaml.safe_load(overlay_path.read_text())
    cfg = {**cfg, **overlay}
    batch, accumulation, global_batch = MODES[args.mode]
    cfg.update(
        per_device_batch_size=batch,
        gradient_accumulation_steps=accumulation,
        target_global_batch_size=global_batch,
        gradient_checkpointing=args.activation_checkpointing == "on",
        output_dir=str(args.run_dir),
    )
    if "resume_from" in cfg:
        cfg["resume_from"] = None
    cfg["hardware_and_compute_budget"] = (
        f"Fresh 4-H100 pilot; {batch}/GPU x 4 x accum{accumulation} = {global_batch}; "
        "BF16 online, FP32 masters; LR G/F 2e-6; 200 successful G / 1000 F; "
        "fit/speed/quality unverified"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write("# Fresh recipient pilot; use the BF16-online/FP32-master backend.\n")
        yaml.safe_dump(cfg, handle, sort_keys=False)
    print(f"Wrote {args.output}; {batch}/GPU x 4 x accum{accumulation} = {global_batch}.")
    print("No training started. Verify FP32 optimizer masters and EMA-from-master before launch.")


if __name__ == "__main__":
    main()
