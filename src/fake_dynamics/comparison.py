"""Read-only manifest audit; matched metadata is not proof of scientific validity."""

import argparse
import json
import math
from pathlib import Path

BOOKKEEPING = {
    "output_dir",
    "generator_checkpoint",
    "guidance_checkpoint",
    "prompts_path",
    "total_generator_updates",
    "checkpoint_interval_g_updates",
    "checkpoint_keep_last",
    "probe_interval_g_updates",
    "sample_interval_g_updates",
    "performance_interval_g_updates",
    "tensorboard",
    "tensorboard_flush_seconds",
    "hardware_and_compute_budget",
}
RESEARCH_FIELDS = {
    "generator_lr",
    "fake_lr",
    "teacher_cfg",
    "teacher_anchor_beta_max",
    "teacher_anchor_start_g_update",
    "teacher_anchor_ramp_g_updates",
    "beta_override",
    "consistency_weight_max",
    "consistency_ramp_g_updates",
    "consistency_override",
    "generator_ema_decay",
}
REQUIRED_CONFIG = RESEARCH_FIELDS | {
    "backend",
    "backbone",
    "backbone_revision",
    "resolution",
    "prediction_type",
    "generator_anchors",
    "adam_betas",
    "weight_decay",
    "max_grad_norm",
    "fake_updates_per_generator",
    "fake_warmup_updates",
    "fake_initialization",
    "fake_timestep_range_inclusive",
    "dm_timestep_range_inclusive",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "mixed_precision",
    "conversion_dtype",
    "seed",
    "teacher_weight_dtype",
    "ema_device",
    "ema_forward_dtype",
    "optimizer_state_sharding",
    "distributed_strategy",
    "gradient_checkpointing",
}


def read_manifest(run):
    root = Path(run)
    if root.is_file():
        return json.loads(root.read_text())
    latest = json.loads((root / "latest.json").read_text())
    return json.loads((Path(latest["checkpoint"]) / "manifest.json").read_text())


def audit(left, right, allowed_fields=(), left_report=None, right_report=None):
    allowed = set(allowed_fields)
    if allowed - RESEARCH_FIELDS:
        raise ValueError(
            f"Only declared research knobs may vary: {sorted(allowed - RESEARCH_FIELDS)}"
        )
    issues, differences, costs = [], [], []

    def compare(name, a, b, permitted=False):
        if a != b:
            differences.append({"field": name, "left": a, "right": b, "declared": permitted})
            if not permitted:
                issues.append(f"Mismatch: {name}")

    for label, manifest in (("left", left), ("right", right)):
        cfg = manifest.get("config", {})
        if REQUIRED_CONFIG - cfg.keys():
            issues.append(f"Missing {label} config fields: {sorted(REQUIRED_CONFIG - cfg.keys())}")
        for key in (
            "config",
            "world_size",
            "global_batch_size",
            "prompt_sha256",
            "scheduler",
            "environment",
            "state",
            "training_sha256",
        ):
            if not manifest.get(key):
                issues.append(f"Missing {label}.{key}; use a checkpoint manifest from v0.3+")
        if cfg.get("backend") != "sdxl" or cfg.get("debug_anchor_cycle"):
            issues.append(f"{label} must be a scientific SDXL run, not toy/debug smoke")
        effective = (
            cfg.get("per_device_batch_size", 0)
            * manifest.get("world_size", 0)
            * cfg.get("gradient_accumulation_steps", 0)
        )
        if effective != manifest.get("global_batch_size"):
            issues.append(f"{label} effective batch is inconsistent")
        environment = manifest.get("environment", {})
        if not environment.get("code_revision") or environment.get("code_dirty") is not False:
            issues.append(f"{label} needs a recorded clean code revision")
        state = manifest.get("state", {})
        k_g, k_f = state.get("generator_updates", 0), state.get("fake_updates", 0)
        if k_g < 1 or state.get("fake_in_cycle", 0) != 0:
            issues.append(f"{label} needs a completed G cycle")
        if k_f != k_g * cfg.get("fake_updates_per_generator", 0) + cfg.get(
            "fake_warmup_updates", 0
        ):
            issues.append(f"{label} F:G counters do not match the declared schedule")
        for network, successful in (("generator", k_g), ("fake", k_f)):
            if state.get(f"{network}_attempts") != successful:
                issues.append(
                    f"{label} has missing attempt counters or skipped {network} windows; data exposure needs review"
                )
        costs.append(
            {
                "run": label,
                "k_G": k_g,
                "k_F": k_f,
                "successful_G_samples": k_g * effective,
                "successful_F_samples": k_f * effective,
                "training_seconds": state.get("elapsed_training_seconds"),
                "training_gpu_hours": manifest.get("world_size", 0)
                * state.get("elapsed_training_seconds", 0)
                / 3600,
                "rank0_forward_counts": state.get("forward_counts"),
            }
        )
        for model in ("generator", "fake"):
            if not manifest.get("provenance", {}).get(model, {}).get("sha256"):
                issues.append(f"Missing {label} initialization hash for {model}")

    a_cfg, b_cfg = left.get("config", {}), right.get("config", {})
    for key in sorted(a_cfg.keys() | b_cfg.keys()):
        if key not in BOOKKEEPING:
            compare(f"config.{key}", a_cfg.get(key), b_cfg.get(key), key in allowed)
    for key in ("world_size", "global_batch_size", "prompt_sha256", "scheduler"):
        compare(key, left.get(key), right.get(key))
    for model in ("generator", "fake"):
        compare(
            f"init.{model}.sha256",
            left.get("provenance", {}).get(model, {}).get("sha256"),
            right.get("provenance", {}).get(model, {}).get("sha256"),
        )
    for key in ("generator_updates", "fake_updates"):
        compare(key, left.get("state", {}).get(key), right.get("state", {}).get(key))
    compare("training_environment", left.get("environment"), right.get("environment"))

    if (left_report is None) != (right_report is None):
        issues.append("Supply both evaluation reports or neither")
    if left_report is not None and right_report is not None:
        for label, report, manifest in (
            ("left", left_report, left),
            ("right", right_report, right),
        ):
            generation = report.get("generation", {})
            if generation.get("checkpoint_sha256") != manifest.get("training_sha256"):
                issues.append(f"{label} evaluation does not belong to this audited checkpoint")
            if generation.get("complete") is not True:
                issues.append(f"{label} evaluation generation is incomplete")
            fid = report.get("metrics", {}).get("fid", {}).get("value")
            if not isinstance(fid, (int, float)) or not math.isfinite(fid):
                issues.append(f"{label} evaluation requires a finite measured FID")
            if not report.get("reference", {}).get("tree_sha256"):
                issues.append(f"Missing {label} evaluation reference hash")
            for name, record in report.get("metrics", {}).items():
                value = record.get("value")
                if value is not None and (
                    type(value) not in (int, float) or not math.isfinite(value)
                ):
                    issues.append(f"Invalid {label} metric: {name}")
        for key in (
            "count",
            "weights",
            "prompts_sha256",
            "seed_base",
            "seed_policy",
            "batch_size",
            "resolution",
            "sampler",
            "external_cfg",
            "watermark",
            "quantization",
            "scheduler",
            "environment",
        ):
            a, b = left_report.get("generation", {}), right_report.get("generation", {})
            if key not in a or key not in b:
                issues.append(f"Missing generation protocol field: {key}")
            compare(f"generation.{key}", a.get(key), b.get(key))
        for key in (
            "upstream_commit",
            "evaluator_sha256",
            "fid_feature_extractor",
            "resize",
            "pilot",
            "environment",
        ):
            if key not in left_report or key not in right_report:
                issues.append(f"Missing evaluation protocol field: {key}")
            compare(f"evaluation.{key}", left_report.get(key), right_report.get(key))
        compare(
            "reference.tree_sha256",
            left_report.get("reference", {}).get("tree_sha256"),
            right_report.get("reference", {}).get("tree_sha256"),
        )
        for name in sorted(
            left_report.get("metrics", {}).keys() | right_report.get("metrics", {}).keys()
        ):
            records = [
                report.get("metrics", {}).get(name, {}) for report in (left_report, right_report)
            ]
            metadata = [
                {k: v for k, v in row.items() if k not in ("value", "display_times_100")}
                for row in records
            ]
            compare(f"metric.{name}.provenance", *metadata)
            measured = [row.get("value") is not None for row in records]
            compare(f"metric.{name}.measured", *measured)

    return {
        "declared_comparison_metadata_matched": not issues,
        "evaluation_audited": left_report is not None and right_report is not None,
        "declared_variable_fields": sorted(allowed),
        "differences": differences,
        "blocking_issues": issues,
        "costs": costs,
        "limitation": "Metadata audit only. Does not prove numerical correctness, equal wall-time, statistical significance or paper reproduction.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--allow-field", action="append", default=[])
    parser.add_argument("--left-report")
    parser.add_argument("--right-report")
    args = parser.parse_args()
    reports = [
        json.loads(Path(p).read_text()) if p else None
        for p in (args.left_report, args.right_report)
    ]
    result = audit(read_manifest(args.left), read_manifest(args.right), args.allow_field, *reports)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(1 if result["blocking_issues"] else 0)


if __name__ == "__main__":
    main()
