# Evaluation and benchmark provenance

Goal from the supplied research spec: FID < 17.80 on a compatible SDXL 4-step
COCO2014 validation 10K protocol. There is no result in this repository. The
supplied Decoupled DMD row and DMD2 zoo values are references, not measured rows.
Read original_handoff_vi.md §§13–15 for metric table, source links and caveats.

## Initialize the metric pipeline on the recipient's machine

```bash
bash scripts/prepare_evaluator.sh
```

Uses a clean upstream DMD2 checkout at
`8d8fa55633d47cfb81bbc7a892e7248f9518763f`. FID calls its `compute_fid` directly
with native uint8 pixels, CenterCropLongEdge and PIL LANCZOS resize to 512. Its
vendored cleanfid feature builder in turn imports parts of installed clean-fid;
that package is pinned to 0.1.35 here and still needs baseline calibration. CLIP
uses the upstream ViT-g-14 / laion2b_s12b_b42k code, caption prefix and token logic.
The install script pins the OpenAI CLIP source commit needed by this path.

Unzip your verified COCO package explicitly into an ignored asset directory.
Inspect the ZIP structure first and preserve `subset` and `all_prompts.pkl`.
Convert the ordered prompt list:

```bash
fdmd convert-prompts --input assets/coco10k/all_prompts.pkl \
  --output assets/coco10k/prompts.jsonl --trusted-pickle
```

Generate initialization **before training** using the same configuration as later
checkpoints:

```bash
fdmd --debug sample --config configs/sdxl.yaml \
  --checkpoint "$GENERATOR_CHECKPOINT" --initial-dmd2 --weights generator \
  --prompts assets/coco10k/prompts.jsonl --output samples/initial-coco10k \
  --seed 10 --count 10000
fdmd --debug evaluate --samples samples/initial-coco10k \
  --reference assets/coco10k/subset --upstream assets/DMD2 \
  --output reports/initial-coco10k --clip
```

Sampling ignores unset training budget fields. Use a checkpoint-directory sample
command from RUNBOOK.md for new G/EMA. The evaluator checks image/mapping hashes,
sample count and upstream revision/dirty state. Native 10K pixels use approximately
31 GB temporary disk in a memory map at 1024, removed when evaluation exits
normally. Check disk and reference count before launching. Other parts of upstream
FID may still allocate features/worker buffers; mmap is not a RAM guarantee.

Pilot evaluation can pass `--expected-count 1000` after generating `--count 1000`.
It is labeled pilot and cannot directly substantiate beating FID-10K 17.80.

## Metric extensions

FID and raw CLIP-S are wired. CLIP raw cosine is recorded separately from an
explicit display ×100 value; the paper's scale must be checked before using it.
ImageReward, HPS v2.1 and HPS v3 remain null unless the recipient supplies verified
local metric plugins. This boundary avoids guessing model versions and score
scales, especially for HPS v3.

Each `--metric-plugin package.module:function` receives keyword arguments
`image_paths`, `prompts`, `device` and returns a dictionary whose supported names
are `image_reward`, `hps_v2_1`, `hps_v3`. Each record must contain:

```json
{
  "value": 0.0,
  "scale": "replace with verified raw/display convention",
  "library_version": "replace with actual version or commit",
  "model_revision": "replace with exact model revision/hash",
  "aggregation": "replace with actual averaging/transform"
}
```

The numeric value above describes the schema only; it is not a benchmark result.
Plugins are explicit recipient code executed locally. Put them in a versioned
module in your experiment branch and include model/protocol evidence. Unknown
names, missing provenance and nonfinite outputs fail. Do not silently multiply
every column by 100.

## Required comparability audit

The manifest preserves checkpoint/weight choice, config/hash, native generation,
sampler, zero external CFG, watermark setting, image and prompt order hashes,
mapping, per-image seeds, reference tree hashes, upstream evaluator hash, library
versions, crop/resize and feature extractor identity. Save actual feature-model
cache hashes and a full `pip freeze` with the report as additional evidence.

The sample seed policy here is `seed + prompt index` with independent initial and
re-noising draws per image. Original DMD2 evaluation uses a batched/global stream;
seed 10 alone does not make them identical. Compare original/native samples and
metrics on the initial checkpoint. Audit conversion precision, caption selection,
image subset, extractor model/cache, installed clean-fid source, Pillow antialias,
resize/crop, OpenCLIP model and generation batch/order before declaring equivalence.

Even a lower number yields `below_target_on_this_protocol=true`, while
`paper_protocol_equivalence=unverified` and `paper_benchmark_win_claim=false`
remain in the automatic report. A scientist can make a justified comparison only
after documenting equivalence separately. The code does not infer it from score.

DMD2 zoo 19.32 versus the supplied paper's DMD2 18.95 discrepancy is unresolved.
Warm start has GAN history; this is continued training without adversarial loss.
Keep initialization and additional compute visible. Do not claim equal-budget
superiority, variance reduction or novelty from a single FID result. Full matched
baseline retraining is not a prerequisite to evaluating the new checkpoint.
