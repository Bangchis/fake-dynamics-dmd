import json
import pickle
from pathlib import Path

from .data import sha256_file
from .runtime import write_json

HUB_REVISION = "be22767697a1f3ca656b73c776e15fa335c86c6c"
PAIRED_PATH = (
    "model/sdxl/sdxl_cond999_8node_lr5e-7_denoising4step_diffusion1000_gan5e-3_"
    "guidance8_noinit_noode_backsim_scratch_checkpoint_model_019000"
)


def fetch_assets(destination, *, coco=False):
    from huggingface_hub import hf_hub_download

    files = [
        f"{PAIRED_PATH}/pytorch_model.bin",
        f"{PAIRED_PATH}/pytorch_model_1.bin",
        "data/laion/captions_laion_score6.25.pkl",
    ]
    if coco:
        files.append("data/coco/coco10k.zip")
    rows = []
    for filename in files:
        path = hf_hub_download(
            "tianweiy/DMD2", filename, revision=HUB_REVISION, local_dir=destination
        )
        rows.append({"file": filename, "local_path": path, "sha256": sha256_file(path)})
    write_json(
        Path(destination) / "asset_manifest.json",
        {"repo": "tianweiy/DMD2", "revision": HUB_REVISION, "files": rows},
    )
    return rows


def convert_prompts(source, output, trusted_pickle=False):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("Output exists; choose a new prompt file")
    if source.suffix == ".pkl":
        if not trusted_pickle:
            raise ValueError(
                "Only convert a trusted pickle: pass --trusted-pickle for verified DMD2 assets"
            )
        with source.open("rb") as stream:
            values = pickle.load(stream)
    elif source.suffix == ".txt":
        values = source.read_text().splitlines()
    else:
        values = json.loads(source.read_text())
    # Upstream SDTextDataset indexes a list of strings. Do not guess dictionary caption selection.
    if not isinstance(values, (list, tuple)) or not all(
        v is None or isinstance(v, str) for v in values
    ):
        raise ValueError(
            "Expected an ordered list of caption strings; inspect other schemas explicitly"
        )
    if not values:
        raise ValueError("No prompts")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for index, prompt in enumerate(values):
            stream.write(
                json.dumps({"id": str(index), "prompt": prompt or ""}, ensure_ascii=False) + "\n"
            )
    write_json(
        output.with_suffix(".provenance.json"),
        {
            "source": str(source),
            "source_sha256": sha256_file(source),
            "output_sha256": sha256_file(output),
            "count": len(values),
            "caption_selection": "preserve original list order, one caption per entry; None becomes empty string",
        },
    )
