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
| Single/DDP gradient accumulation; explicit zero1 option | Loss/count mean, DDP no_sync until last microbatch, clip/step/EMA once. H100 effective batch 128; no FSDP/LoRA. Runtime validation is pending. |
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
- Superseded initial batch 1/GPU/global 2 plan: see the owner-requested update below.
- CD maximum screen 0.03/0.1/0.3 at 300 G, unchanged beta ramp; winner to 1500 G
  before judging full anchoring. Optional control beta=0/CD=0 at 1500 G. Resume
  justified winner to 5000 total G. No automated quality winner or early-stop rule.
- Source handoff remains verbatim. No Mac tests, GPU runtime or metrics executed.

## Owner LR and fair-batch update (2026-10-08)

The owner explicitly chose LR 1e-6 and rejected global batch 2 as too small for
comparison. Both G/F now use constant 1e-6 (paper reference 5e-7). H100 target is
effective batch 128: physical 2 × 2 ranks × accumulation 32. If measured VRAM
requires physical 1, use accumulation 64 in a new run; retain effective batch/LR.

Accumulation is implemented for both F and G, not merely enabled in YAML. Every
window keeps model weights, beta/CD and a shared anchor fixed. Loss gradients are
averaged; unscale/clip/step/EMA/counters execute once at the window boundary.
Nonfinite loss/gradients reject the entire window, clear its gradients and keep
G/EMA counters unchanged. Checkpoints never save a partial gradient window.
DDP invalid-loss backward completes reducer hooks before gradients are discarded.

Short smoke/resume uses accumulation 2; a separate four-anchor batch-smoke uses
full effective 128. Scientific stages reject effective-batch mismatch. Resume
requires unchanged physical batch and accumulation, even if their product matches.
Control matches 1500 G and optionally 5000 G for final comparison. Added numerical
mean/clipping/EMA/skip tests and two-rank mean-gradient/resume tests are authored
but unexecuted. Runtime correctness and memory fit await recipient evidence.

At the owner's further request to check fair comparisons, v0.3 checkpoint JSON
manifests include current config, topology/effective batch, prompt hash, scheduler
and environment. A read-only comparator audits completed run budgets, declared
research differences, initial/data/code hashes and optional paired evaluation
protocols without loading models. Skips require data-exposure review. GPU cost is
reported, not forced equal. This metadata audit cannot certify statistical or
numerical validity or comparison to external paper scores. Tests remain unrun.
