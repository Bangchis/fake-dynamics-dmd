#!/usr/bin/env bash
# RECIPIENT MACHINE ONLY. Does not install dependencies or fetch assets/metric weights.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GENERATOR_CHECKPOINT:?Set the full paired DMD2 generator checkpoint path}"
: "${GUIDANCE_CHECKPOINT:?Set the matching DMD2 guidance checkpoint path}"
: "${TRAIN_PROMPTS_JSONL:?Set the converted LAION training prompts JSONL path}"
for input in "$GENERATOR_CHECKPOINT" "$GUIDANCE_CHECKPOINT" "$TRAIN_PROMPTS_JSONL"; do
  [[ -f "$input" ]] || { echo "Missing input file: $input" >&2; exit 1; }
done
stage="${1:-pilot}"
candidate="${2:-cd-default}"
case "$candidate" in
  cd-low) cd_weight=0.03 ;;
  cd-default) cd_weight=0.1 ;;
  cd-high) cd_weight=0.3 ;;
  *) echo 'Candidate must be cd-low, cd-default or cd-high' >&2; exit 1 ;;
esac
root="${RUN_ROOT:-runs/h100}"
run_dir="$root/$candidate"
extra=()
resume=()
# Scientific stages always target paper effective batch 128. Physical batch must
# be measured on H100; lowering it increases accumulation instead of changing LR.
batch="${PER_DEVICE_BATCH_SIZE:-2}"
[[ "$batch" =~ ^[1-9][0-9]*$ ]] || { echo 'Batch must be a positive integer' >&2; exit 1; }
(( 128 % (2 * batch) == 0 )) || { echo '2 * batch must divide 128' >&2; exit 1; }
accumulation=$((128 / (2 * batch)))
target_batch=128
case "$stage" in
  smoke)
    run_dir="$root/smoke"
    budget=8
    accumulation=2
    target_batch=$((2 * batch * accumulation))
    extra+=(--set debug_anchor_cycle=true --set beta_override=0.05 --set consistency_override=0.1
      --set checkpoint_interval_g_updates=4 --set probe_interval_g_updates=1
      --set sample_interval_g_updates=4 --set performance_interval_g_updates=1)
    ;;
  smoke-resume)
    run_dir="$root/smoke"
    budget=12
    accumulation=2
    target_batch=$((2 * batch * accumulation))
    extra+=(--set debug_anchor_cycle=true --set beta_override=0.05 --set consistency_override=0.1
      --set checkpoint_interval_g_updates=4 --set probe_interval_g_updates=1
      --set sample_interval_g_updates=4 --set performance_interval_g_updates=1)
    ;;
  batch-smoke)
    run_dir="$root/batch-smoke"
    budget=4
    extra+=(--set debug_anchor_cycle=true --set beta_override=0.05 --set consistency_override=0.1
      --set checkpoint_interval_g_updates=4 --set probe_interval_g_updates=1
      --set sample_interval_g_updates=4 --set performance_interval_g_updates=1)
    ;;
  screen)
    budget=300
    extra+=(--set checkpoint_interval_g_updates=300 --set sample_interval_g_updates=100)
    ;;
  pilot) budget=1500 ;;
  main) budget=5000 ;;
  baseline)
    run_dir="$root/control"
    budget=1500
    extra+=(--set beta_override=0.0 --set consistency_override=0.0)
    ;;
  baseline-main)
    run_dir="$root/control"
    budget=5000
    extra+=(--set beta_override=0.0 --set consistency_override=0.0)
    ;;
  *) echo 'Stage: smoke | smoke-resume | batch-smoke | screen | pilot | main | baseline | baseline-main' >&2; exit 1 ;;
esac
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  resume=(--resume "$RESUME_CHECKPOINT")
elif [[ -f "$run_dir/latest.json" ]]; then
  checkpoint="$(python - "$run_dir/latest.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['checkpoint'])
PY
)"
  resume=(--resume "$checkpoint")
elif [[ "$stage" == main || "$stage" == baseline-main || "$stage" == smoke-resume ]]; then
  echo "Stage $stage needs a completed checkpoint in $run_dir or RESUME_CHECKPOINT" >&2
  exit 1
fi
# Resume requires identical physical batch AND accumulation, not merely equal effective batch.
printf 'Stage=%s candidate=%s batch/GPU=%s accumulation=%s effective_batch=%s total_G=%s output=%s\n' "$stage" "$candidate" "$batch" "$accumulation" "$target_batch" "$budget" "$run_dir"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m fake_dynamics.cli --debug train \
  --config configs/h100_2x80.yaml \
  --set "generator_checkpoint=$GENERATOR_CHECKPOINT" \
  --set "guidance_checkpoint=$GUIDANCE_CHECKPOINT" \
  --set "prompts_path=$TRAIN_PROMPTS_JSONL" \
  --set "output_dir=$run_dir" --set "per_device_batch_size=$batch" \
  --set "gradient_accumulation_steps=$accumulation" --set "target_global_batch_size=$target_batch" \
  --set "total_generator_updates=$budget" --set "consistency_weight_max=$cd_weight" \
  "${extra[@]}" "${resume[@]}"
