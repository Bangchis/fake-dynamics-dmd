"""Bridge to inspected DMD2 FID/CLIP implementation; preserve protocol uncertainty."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .data import sha256_file
from .runtime import environment_manifest, write_json

DMD2_COMMIT = "8d8fa55633d47cfb81bbc7a892e7248f9518763f"
METRIC_NAMES = ("image_reward", "hps_v2_1", "hps_v3")


def tree_manifest(directory):
    root = Path(directory)
    rows = [
        {"file": str(path.relative_to(root)), "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    if not rows:
        raise ValueError("Reference directory is empty")
    import hashlib

    return {
        "file_count": len(rows),
        "image_count": sum(
            Path(row["file"]).suffix.lower()
            in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
            for row in rows
        ),
        "tree_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "files": rows,
    }


def evaluate(
    samples_dir,
    reference_dir,
    upstream_path,
    output_dir,
    *,
    clip=False,
    metric_plugins=None,
    expected_count=10000,
    device="cuda",
):
    from PIL import Image

    source = Path(samples_dir)
    generation = json.loads((source / "generation_manifest.json").read_text())
    if not generation.get("complete"):
        raise ValueError("Generation is incomplete")
    if sha256_file(source / "mapping.jsonl") != generation["mapping_sha256"]:
        raise ValueError("Generation mapping checksum mismatch")
    rows = [json.loads(line) for line in (source / "mapping.jsonl").read_text().splitlines()]
    if len(rows) != expected_count or generation["count"] != expected_count:
        raise ValueError(f"Expected {expected_count} images; explicitly set a smaller pilot count")
    refs = tree_manifest(reference_dir)
    if expected_count == 10000 and refs["image_count"] != 10000:
        raise ValueError("Full COCO10K evaluation requires 10000 reference images")
    upstream = Path(upstream_path).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain"], text=True
    ).strip()
    if revision != DMD2_COMMIT or dirty:
        raise ValueError("Evaluator requires a clean DMD2 checkout at the pinned commit")
    sys.path.insert(0, str(upstream))
    from main.coco_eval.coco_evaluator import compute_clip_score, compute_fid

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose an empty evaluation output directory")
    r = generation["resolution"]
    # Native pixels are passed to upstream's exact center-crop/LANCZOS path.
    # A 10K run uses ~31 GB scratch disk at 1024. Never keep the entire array in RAM.
    scratch = output / "native_pixels.memmap"
    images = np.memmap(scratch, mode="w+", dtype=np.uint8, shape=(len(rows), r, r, 3))
    try:
        for index, row in enumerate(rows):
            path = source / row["file"]
            if path.parent.resolve() != source.resolve() or sha256_file(path) != row["sha256"]:
                raise ValueError(f"Invalid image file/checksum: {row['file']}")
            with Image.open(path) as image:
                if image.size != (r, r):
                    raise ValueError("Generated resolution differs from manifest")
                images[index] = np.asarray(image.convert("RGB"))
        images.flush()
        fid = float(compute_fid(images, reference_dir, device, resize_size=512))
        if not np.isfinite(fid):
            raise ValueError("FID evaluator returned a nonfinite value")
        metrics = {
            "fid": {"value": fid, "scale": "raw", "source": "pinned_DMD2_evaluator"},
            "clip_s": {"value": None, "status": "not_requested"},
        }
        if clip:
            raw = float(
                compute_clip_score(
                    images,
                    [row["prompt"] for row in rows],
                    clip_model="ViT-G/14",
                    device=device,
                    how_many=len(rows),
                )
            )
            metrics["clip_s"] = {
                "value": raw,
                "scale": "raw_cosine",
                "display_times_100": raw * 100,
                "model": "ViT-g-14/laion2b_s12b_b42k",
                "source": "pinned_DMD2_evaluator",
            }
        for name in METRIC_NAMES:
            metrics[name] = {"value": None, "status": "plugin_required_protocol_unverified"}
        for specification in metric_plugins or []:
            # Explicit local plugin API, no assumed package/model/metric scaling.
            module_name, separator, function_name = specification.partition(":")
            if not separator:
                raise ValueError("Metric plugin must be MODULE:FUNCTION")
            function = getattr(importlib.import_module(module_name), function_name)
            result = function(
                image_paths=[str(source / row["file"]) for row in rows],
                prompts=[row["prompt"] for row in rows],
                device=device,
            )
            for name, record in result.items():
                if name not in METRIC_NAMES:
                    raise ValueError(f"Unsupported extra metric: {name}")
                required = {"value", "scale", "library_version", "model_revision", "aggregation"}
                if not required <= record.keys() or not np.isfinite(float(record["value"])):
                    raise ValueError(f"Incomplete metric provenance: {name}")
                metrics[name] = {**record, "plugin": specification}
        manifest = {
            "generation": generation,
            "metrics": metrics,
            "reference": refs,
            "upstream_commit": revision,
            "evaluator_sha256": sha256_file(upstream / "main/coco_eval/coco_evaluator.py"),
            "fid_feature_extractor": "DMD2 vendored cleanfid inception_v3",
            "resize": "DMD2 CenterCropLongEdge then PIL LANCZOS 512x512",
            "environment": environment_manifest(),
            "pilot": expected_count != 10000,
            "paper_protocol_equivalence": "unverified",
            "target_fid": 17.80,
            "below_target_on_this_protocol": fid < 17.80,
            "paper_benchmark_win_claim": False,
        }
        write_json(output / "evaluation_manifest.json", manifest)
        return manifest
    finally:
        del images
        scratch.unlink(missing_ok=True)
