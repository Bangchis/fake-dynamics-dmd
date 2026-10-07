import argparse
import json
import sys

from .config import Config


def build_parser():
    parser = argparse.ArgumentParser(prog="fdmd")
    parser.add_argument("--debug", action="store_true", help="Show full exception traceback")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "doctor", "train", "sample"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
        if name == "train":
            sub.add_argument("--resume")
            sub.add_argument("--continue-after-warmup", action="store_true")
        if name == "sample":
            sub.add_argument("--checkpoint", required=True)
            sub.add_argument("--prompts", required=True)
            sub.add_argument("--output", required=True)
            sub.add_argument("--weights", choices=["generator", "ema"], default="ema")
            sub.add_argument("--initial-dmd2", action="store_true")
            sub.add_argument("--seed", type=int, default=10)
            sub.add_argument("--count", type=int)
    inspect = commands.add_parser("inspect-checkpoint")
    inspect.add_argument("path")
    fetch = commands.add_parser("fetch-assets")
    fetch.add_argument("--output", default="assets")
    fetch.add_argument("--include-coco", action="store_true")
    convert = commands.add_parser("convert-prompts")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--trusted-pickle", action="store_true")
    export = commands.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--weights", choices=["generator", "ema"], required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--samples", required=True)
    evaluation.add_argument("--reference", required=True)
    evaluation.add_argument("--upstream", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--clip", action="store_true")
    evaluation.add_argument("--metric-plugin", action="append", default=[])
    evaluation.add_argument("--expected-count", type=int, default=10000)
    return parser


def dispatch(args):
    config = Config.load(args.config, args.set) if hasattr(args, "config") else None
    if args.command == "plan":
        return {
            "config": config.as_dict(),
            "config_hash": config.digest(),
            "missing_runtime_fields": config.missing_runtime_fields(),
            "schedule": {
                k: dict(zip(("beta", "lambda_cd"), config.weights(k)))
                for k in (0, 100, 101, 1100, 2000)
            },
            "status": "implementation_only_no_SDXL_smoke_or_benchmark",
        }
    if args.command == "doctor":
        import torch

        from .runtime import environment_manifest

        return {
            "environment": environment_manifest(),
            "cuda_available": torch.cuda.is_available(),
            "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            "missing_runtime_fields": config.missing_runtime_fields(),
            "model_loaded": False,
            "training_executed": False,
        }
    if args.command == "train":
        from .trainer import Trainer

        trainer = Trainer(config, args.resume)
        trainer.run(args.continue_after_warmup)
        return {"output_dir": config.output_dir, "state": trainer.state}
    if args.command == "sample":
        from .inference import generate

        return generate(
            config,
            args.checkpoint,
            args.prompts,
            args.output,
            weights=args.weights,
            seed=args.seed,
            count=args.count,
            initial_dmd2=args.initial_dmd2,
        )
    if args.command == "inspect-checkpoint":
        from .weights import inspect_weights

        return inspect_weights(args.path)
    if args.command == "fetch-assets":
        from .assets import fetch_assets

        return fetch_assets(args.output, coco=args.include_coco)
    if args.command == "convert-prompts":
        from .assets import convert_prompts

        convert_prompts(args.input, args.output, args.trusted_pickle)
        return {"output": args.output}
    if args.command == "export":
        from .inference import export_unet

        export_unet(args.checkpoint, args.output, args.weights)
        return {"output": args.output}
    if args.command == "evaluate":
        from .evaluation import evaluate

        result = evaluate(
            args.samples,
            args.reference,
            args.upstream,
            args.output,
            clip=args.clip,
            metric_plugins=args.metric_plugin,
            expected_count=args.expected_count,
        )
        return {
            "output": args.output,
            "metrics": result["metrics"],
            "paper_protocol_equivalence": result["paper_protocol_equivalence"],
        }
    raise ValueError("Unknown command")


def main():
    args = build_parser().parse_args()
    try:
        result = dispatch(args)
    except Exception as error:
        if args.debug:
            raise
        print(
            f"fdmd: {type(error).__name__}: {error}\nUse fdmd --debug for traceback.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    if result is not None:
        print(json.dumps(result, indent=2, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
