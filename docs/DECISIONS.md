# Implementation decisions

2026-10-08 initial implementation, based on the supplied handoff.

| Decision | Reason and consequence |
|---|---|
| New package rather than fork old trainer | Original trainer couples denoising and LMDB/GAN loaders. A prompt-only loop keeps the specified four-step sampler and isolates math for review. No claim of patching the original files. |
| DMD2 commit 8d8fa55 is the code reference | Matches the supplied inspected revision; generator/backsimulation/conditioning/metric semantics were read from those actual sources. |
| SDXL/DDIM scheduler loaded by revision | Uses native checkpoint alphas; fail on prediction-type/timestep mismatch. Helper precision defaults to float64 as in upstream training. |
| Proposed branchwise CFG-reference normalization | Transparent adapter from handoff §5.3; not verified official SDXL Decoupled code. DM numerator conditional but its scale still depends on teacher CFG. |
| Conditional/unconditional T calls made separately | Same states/time IDs, zero unconditional embeddings. Avoid a doubled teacher batch; extra calls are real costs. |
| 5 successful F then 1 successful G | Follows supplied pseudocode; differs in interleaving from original outer-loop implementation. Beta fixed per cycle, skip retries keep intended success counts. |
| Fresh constant-LR AdamW | No inherited source optimizer/outer-loop LR scheduler. No 5× unit mismatch; any future G/F-unit LR schedule must be saved/restored. |
| FP32 G/F and EMA master; optional network autocast | Small optimizer/EMA updates stay FP32. H100 mode uses BF16 teacher/staged EMA forwards, disclosed as a numerical adaptation. |
| Single/DDP, accumulation 1; explicit zero1 option | H100 mode shares optimizer states across DDP ranks using native Torch 2.6. No FSDP/LoRA/accumulation. Runtime validation is pending. |
| Teacher fallback pauses after fake warmup | Fixed number alone does not prove fitting adequacy; recipient inspects diagnostics before continuing. Paired F remains the default. |
| Nonfinite gradients are exposed | No latent clipping/thresholding/nan_to_num masking. Log AMP skips, preserve failure checkpoint, abort repeated failures. |
| Atomic schema-2 rank-local checkpoints with opt-in retention | Latest replaces only owned completed periodic/final directories; initial assets, selected exports, `.pin` and failure/warmup states remain protected. Resume requires same prompt contents, batch, world size, Torch build and named parameter partition. |
| Per-prompt independent sample seed | Clear stable mapping/re-noising sequence. Differs from original batched evaluator RNG and is disclosed as a protocol discrepancy. |
| Pinned upstream FID/CLIP bridge | Reuses inspected metric functions. External clean-fid components/feature caches still require calibration; other metric versions/scales use explicit plugins. |
| No testing in the authoring session | Owner explicitly stated the Mac CPU is small and requested no tests. Only lightweight static syntax/lint/format checks are permitted here; all execution evidence awaits recipient. |

Future scientific decisions should record trigger/evidence, changed config/code,
expected behavior, verification status and benchmark implications. Do not quietly
add GAN, boundary CD at 0, shared weights, teacher trajectory CD, EMA-F or an
adaptive controller. Diagnostic ablations use explicit config overrides and a
new run, preserving the full method's provenance.

## 2 H100 recipient update (2026-10-08)

The owner requested detailed logs/TensorBoard and minimal saving, with existing
metric installations reused. This later request authorizes bounded deletion of
managed rolling checkpoints; it does not authorize discarding initialization or
a chosen best model. H100_RUN_PLAN.md and CONFIG_SOURCES.md specify the experiment.

- CPU FP32 EMA master, functional BF16 CUDA snapshot, released before backward
  and invalidated on update/resume; teacher stored BF16. Compare numerical targets
  against the FP32 reference on real SDXL before concluding parity.
- Native stage-1 optimizer state sharing, DDP bucket views and AdamW foreach=false
  reduce residency/temporary peaks. Each rank persists its local optimizer/RNG
  state directly; no multi-GB CUDA object consolidation.
- TensorBoard uses rank-aggregated numeric diagnostics and rank-0 fixed images.
  Fixed image decoding preserves train RNG and releases VAE. JSONL remains per-rank.
- Batch 1/GPU is an unmeasured starting point. Paper global batch 128 is not matched.
  Keep LR fixed initially, then measure before any batch/LR changes.
- CD maximum screen 0.03/0.1/0.3 at 300 G, unchanged beta ramp; winner to 1500 G
  before judging full anchoring. Optional control beta=0/CD=0 at 1500 G. Resume
  justified winner to 5000 total G. No automated quality winner or early-stop rule.
- Source handoff remains verbatim. No Mac tests, GPU runtime or metrics executed.
