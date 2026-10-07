# Recipient runbook

All commands here are for the recipient's machine. None were run on the author's
Mac. Use the repository root as your working directory. See README for dependency
installation; choose a Torch CUDA wheel matching your driver/GPU. Do not install
upstream DMD2's whole older training environment into this package environment.

## 1. Record and inspect before allocation

```bash
fdmd plan --config configs/sdxl.yaml
fdmd doctor --config configs/sdxl.yaml
python -m pip freeze > environment-freeze.txt
nvidia-smi > nvidia-smi.txt
```

Keep these environment records with your experiment, not in the source commit.
Check host RAM and disk as well as VRAM. Full G/F, optimizer state and FP32 EMA are
large; DDP replicates them. Checkpoint writing/verification and evaluation consume
substantial I/O. No memory or hardware capacity has been measured by the author.

## 2. Fetch verified-source assets explicitly

```bash
fdmd fetch-assets --output assets --include-coco
```

Downloads the two **full paired** DMD2 019000 `.bin` files, LAION captions and
optionally COCO ZIP from pinned HF revision
`be22767697a1f3ca656b73c776e15fa335c86c6c`. This is a large network/disk operation.
Weights/data are gitignored. For gated resources, authenticate on your own machine;
do not commit a token or add it to configs/logs. Teacher components use pinned SDXL
revision `462165984030d82259a11f4367a4eed129e94a7b`.

Set convenient shell paths from the download manifest:

```bash
export GENERATOR_CHECKPOINT=/absolute/path/to/assets/model/sdxl/.../pytorch_model.bin
export GUIDANCE_CHECKPOINT=/absolute/path/to/assets/model/sdxl/.../pytorch_model_1.bin
fdmd inspect-checkpoint "$GENERATOR_CHECKPOINT"
fdmd inspect-checkpoint "$GUIDANCE_CHECKPOINT"
```

Replace `...` with the actual long checkpoint directory from `asset_manifest.json`.
Generator is a raw UNet state dict; guidance should contain `fake_unet.*`. Import
does exact keys and tensor shapes before `strict=True` loading. It ignores all
other guidance keys instead of loading teacher/discriminator heads into F.
Restricted tensor checkpoint loads use `weights_only=True`; unsupported legacy
pickle formats fail instead of silently weakening the loader.

Convert the trusted source captions to the package's prompt-only JSONL:

```bash
fdmd convert-prompts --input assets/data/laion/captions_laion_score6.25.pkl \
  --output assets/train_prompts.jsonl --trusted-pickle
```

Conversion expects an ordered list of strings (None becomes an empty prompt as in
DMD2). Other schemas fail for explicit inspection. Each row has `id` and `prompt`.
Preserve list order; conversion writes source/output hashes. Never use COCO eval
images/captions as the training data for this benchmark.

## 3. Checks and smoke

```bash
pytest -q
fdmd --debug train --config configs/toy.yaml
bash scripts/smoke_gpu.sh
```

Run these only on a suitable recipient machine. The GPU script needs the exported
checkpoint paths and an empty `SMOKE_OUTPUT_DIR` (default `runs/sdxl-smoke`). It
uses batch 1, two G updates, beta/CD overrides and gradient checkpointing. Overrides
exercise nonzero branches; they are not quality hyperparameter recommendations.
Inspect JSONL `cd_active`: random anchors may skip CD at 249. Extend the smoke on
your machine until each desired anchor path is seen; two updates do not guarantee
coverage of all three CD pairs.

```bash
tail -f runs/sdxl-smoke/metrics_rank0.jsonl
```

Pick a completed checkpoint directory under `checkpoints/` and resume into a new
output directory, extending the budget. Pass the exact same smoke overrides,
checkpoint paths, prompts, precision, batch and world size. Example:

```bash
fdmd --debug train --config configs/sdxl.yaml \
  --resume runs/sdxl-smoke/checkpoints/g0000001_f0000005_periodic \
  --set "generator_checkpoint=$GENERATOR_CHECKPOINT" \
  --set "guidance_checkpoint=$GUIDANCE_CHECKPOINT" \
  --set prompts_path=examples/prompts.jsonl \
  --set output_dir=runs/sdxl-smoke-resume \
  --set per_device_batch_size=1 --set total_generator_updates=3 \
  --set checkpoint_interval_g_updates=1 --set probe_interval_g_updates=1 \
  --set gradient_checkpointing=true \
  --set beta_override=0.05 --set consistency_override=0.1
```

The example path is for paired initialization with no fake warmup. Use the actual
directory emitted by your run. Resume restores source k_G/k_F and cycle progress,
not checkpoint label 19000. Teacher/text weights are reloaded by revision. The
same next CPU update should be exact in unit checks; real CUDA replay needs your
chosen numerical tolerance/kernel determinism policy and environment evidence.

## 4. Pilot/quality config

Copy `configs/sdxl.yaml` to gitignored `configs/local-pilot.yaml`. Set actual paths,
batch, total successful G updates, checkpoint/probe intervals, strategy, precision,
gradient checkpointing and hardware budget. Keep beta/CD overrides **null** for
the proposed method. A run config records the actual global batch as per-device
batch × world size. The runtime rejects gradient accumulation other than 1.

```bash
fdmd plan --config configs/local-pilot.yaml
fdmd --debug train --config configs/local-pilot.yaml
```

For measured multi-GPU capacity, set `distributed_strategy: ddp` and launch:

```bash
torchrun --standalone --nproc_per_node=4 -m fake_dynamics.cli --debug train \
  --config configs/local-pilot.yaml
```

The GPU count is an example, not assumed available capacity. Multi-node launch
uses normal torchrun rank/rendezvous environment. Verify DDP on a small workload
before scaling; NCCL/overflow behavior was not run during authoring.

Warm-start fallback: if matching F cannot be used, explicitly set
`fake_initialization: teacher`, `guidance_checkpoint: null` and
`fake_warmup_updates` to an operator-chosen count. The run first freezes G for
fake fitting with beta=0, saves `warmup_complete`, and stops. Inspect tracking and
fixed probes before resuming with `--continue-after-warmup`. A nominal 200–500
updates is only a starting check, not evidence that F has fit. Paired mode needs
no such fallback by default.

## 5. Samples/export

```bash
fdmd --debug sample --config configs/local-pilot.yaml \
  --checkpoint runs/main/checkpoints/ACTUAL_CHECKPOINT_DIRECTORY \
  --weights ema --prompts examples/prompts.jsonl --output samples/fixed-ema --seed 10
fdmd export --checkpoint runs/main/checkpoints/ACTUAL_CHECKPOINT_DIRECTORY \
  --weights ema --output reports/generator_ema.safetensors
```

Choose G or EMA once per evaluation and record it. The sample command runs the
manual DMD2-style four-anchor stochastic sampler, conditional G only, with no
pipeline external CFG or watermark. Seed policy is one independent seed per
prompt index, stable with prompt order. For initialization evaluation use
`--checkpoint "$GENERATOR_CHECKPOINT" --initial-dmd2 --weights generator`.

Export strips only this package's `unet.` wrapper and writes native Diffusers
UNet keys. It is a deployable weight file plus config/provenance sidecar, not a
full pipeline or a training-resume checkpoint. No pretrained weights are pushed
to this source repo.
