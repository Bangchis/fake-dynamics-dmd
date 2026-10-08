# Recipient handoff

Owner request: package a suitable implementation, preserve enough research and
engineering context for a third party to implement/debug it, and publish a public
GitHub repository. The owner has no target environment/GPU available and explicitly
requested **no tests on the author's small Mac CPU**. Authoring date: 2026-10-08.

This is implementation delivery, not a trained model delivery. The original file
`original_handoff_vi.md` is preserved as supplied. Its requested hardware checks,
GPU runs and benchmark are future recipient work, not completed actions.

## Start here

The recipient now has HF access and 2 H100 80GB GPUs. Use H100_RUN_PLAN.md and
CONFIG_SOURCES.md for the chosen hardware mode, bounded CD screen, TensorBoard,
retention and reuse of existing metric installations. These additions are code
delivery only: the Mac no-test constraint remains in force.

1. Read METHOD.md and DECISIONS.md. Keep the distinction between fixed v0 choices,
   proposed hyperparameters, adapter choices and unverified benchmark details.
2. Follow RUNBOOK.md on your machine. Record Python/CUDA/driver/dependency versions,
   available VRAM/host RAM and disk capacity before loading full checkpoints.
3. Run static/unit/toy checks, inspect actual G/F checkpoint keys/shapes, then SDXL
   smoke and a real save/resume check. See VERIFICATION.md for evidence to save.
4. Evaluate initialization with the same chosen sample/metric protocol before
   continuation training. Fix benchmark comparability in EVALUATION.md.
5. Choose batch, strategy and total G updates from measured resources. The beta
   ramp reaches its maximum only at k_G=1100; leave a meaningful post-ramp phase.

## What is ready to review

The package has executable training/inference/evaluation entrypoints, rather than
an empty integration template. Numerical kernels are separate from the SDXL backend
so a maintainer can isolate sampler, signs, coefficient broadcasts and detach
boundaries. Training consumes prompt JSONL directly and never builds the original
LMDB real-image loaders. No discriminator is loaded into either optimizer.

Native DDP and AMP handling are implemented but unexecuted. Checkpoint schema 2
stores G/F/EMA, AdamW, AMP scalers, counters, warmup/cycle progress, per-rank RNG and
prompt-stream states, fixed-probe history, scheduler coefficients/config and
initialization provenance. Optimizer/RNG/data state is written separately by each
rank; shared G/F/EMA weights are written once. V1 model reading remains supported.
Teacher/text encoders reload from the pinned backbone. Bounded retention is opt-in;
the H100 config keeps one latest and protects `.pin`, failure/warmup and exports.

Full SDXL is intentionally full-weight training. The H100 mode implements native
optimizer stage-1 sharing and a FP32 CPU EMA master with a staged BF16 CUDA target
copy. Plain DDP still replicates all model/optimizer state; the explicit zero1 mode
partitions AdamW state only. Gradient accumulation is implemented for G/F; H100
target is effective batch 128 and LR 5e-7 for both optimizers. No FSDP/LoRA. H100 memory
fit, functional EMA forwards and local optimizer resume still require execution.

## First integration uncertainties to resolve

- Actual checkpoint keys, tensor shapes and paired-F extraction have not been
  inspected on downloaded weights. `inspect-checkpoint` and strict import expose
  failures without weakening key matching.
- Installed SDXL/Diffusers forwards, gradient checkpointing and precision have not
  run with real weights. Top-level dependency versions are pinned; transitive
  versions and hardware behavior must be recorded from the recipient environment.
- Real save/resume, AMP overflow and multi-rank paths need execution. Unit tests
  cover intended CPU behavior; no passing result is asserted here.
- FID/CLIP bridge uses pinned DMD2 metric code. Seed policy is deliberately explicit
  and differs from its original batched stream. Feature-weight caches/resize/library
  versions still need protocol audit before comparing to the paper.
- ImageReward/HPS v2.1/HPS v3 require recipient metric plugins with verified models,
  versions, aggregation and scales. Missing metrics are null with a reason.
- Fitting adequacy, critic lag, CD dynamics, stability, GPU cost and final quality
  are research questions. No trained checkpoints or scores are included.

## Return package after a successful run

Deliver the commit SHA, resolved config/hash, environment freeze, asset hashes,
JSONL logs for every rank, initial/pilot/final evaluation manifests, G and EMA weight
choice, retained checkpoints, fixed prompt/seed samples, resource measurements and
the completed verification checklist. Report FID, CLIP-S, ImageReward, HPS v2.1 and
HPS v3 with each column's scaling. Keep absent values absent rather than inventing
numbers or copying the paper's row as a measured result.
