#!/usr/bin/env bash
# RECIPIENT ONLY. Requires an existing CUDA environment and actual paired checkpoint.
set -euo pipefail
: "${GENERATOR_CHECKPOINT:?Set full DMD2 pytorch_model.bin path}"
: "${GUIDANCE_CHECKPOINT:?Set matching DMD2 pytorch_model_1.bin path}"
task_run="${SMOKE_OUTPUT_DIR:-runs/sdxl-smoke}"
if [[ -e "$task_run" ]]; then
  echo "Choose a new SMOKE_OUTPUT_DIR; existing runs are preserved." >&2
  exit 1
fi
fdmd doctor --config configs/sdxl.yaml
fdmd --debug train --config configs/sdxl.yaml \
  --set "generator_checkpoint=$GENERATOR_CHECKPOINT" \
  --set "guidance_checkpoint=$GUIDANCE_CHECKPOINT" \
  --set prompts_path=examples/prompts.jsonl \
  --set "output_dir=$task_run" \
  --set per_device_batch_size=1 \
  --set total_generator_updates=2 \
  --set checkpoint_interval_g_updates=1 \
  --set probe_interval_g_updates=1 \
  --set gradient_checkpointing=true \
  --set beta_override=0.05 \
  --set consistency_override=0.1
echo "Inspect metrics_rank0.jsonl; then resume from a saved g0000001 checkpoint."
echo "Branch overrides are implementation checks; this run is not a quality experiment."
