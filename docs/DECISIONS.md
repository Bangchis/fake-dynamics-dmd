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

## Owner LR and fair-batch update (2026-10-08; LR superseded below)

The owner explicitly chose LR 1e-6 and rejected global batch 2 as too small for
comparison. That revision used constant 1e-6 for G/F (paper reference 5e-7). H100 target is
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

## Match the reference learning rate (2026-10-08; superseded below)

The owner's latest request supersedes the earlier 1e-6 choice: use the original
DMD2 SDXL launch setting, generator_lr=5e-7 and guidance/fake_lr=5e-7. Package
defaults, generic SDXL config, H100 config and fixed sweep settings now agree.
Keep constant LR, effective batch 128, AdamW settings, F:G ratio and ramps as
documented. Candidate/control runs must use the same LR. Matching LR/batch still
does not establish paper reproduction because loss/GAN/initialization differ.

Existing runs retain their recorded LR. Strict resume rejects changing LR from
1e-6 to 5e-7; the reference-setting experiment starts a new run from the same
paired initialization. Only lightweight static checks were performed on the Mac.

## Owner-selected pilot learning rate (2026-10-09; current)

The owner now requests LR 2e-6 for the new pilot. Both generator_lr and fake_lr
are set to 2e-6 in package defaults, SDXL/H100 templates and the fixed sweep
settings. This is four times the original DMD2 reference LR 5e-7; retain that
reference and the original handoff verbatim. LR is constant with fresh AdamW.

The intent is stronger updates during a short continuation pilot, not faster
execution of each cycle. No convergence, stability or benchmark improvement has
been measured at this LR. Candidate/control must use the same declared LR.
Existing runs keep their recorded config; the new LR starts a new run from the
paired DMD2 initialization, not a strict resume with a changed optimizer setting.

The recipient's 4-GPU BF16-online/FP32-master/offload changes remain local and
unmerged. They must set both LR fields in the actual resolved pilot config;
changing these upstream templates does not modify their running process. This
commit changes only LR and its documentation, not batch, beta/CD ramps, precision,
optimizer placement or training budget. The comparison test's mismatched-LR case
now derives a different value from the baseline so it remains meaningful when
defaults change. No tests or training were executed on the author's Mac.

## Short BF16 pilot on the recipient's 4-GPU backend (2026-10-09)

The owner clarified that the goal is a lighter BF16 pilot with LR2e-6, not just
changing LR in the old global128/1500G plan. The recipient reports BF16 online
G/F with FP32 optimizer masters/moments, but their patch/new source files are
unavailable here. No new BF16 optimizer implementation is inferred or merged.

Added a recipient-only overlay and a YAML/JSON config helper preserving local
backend fields/asset paths. It requires an existing train_weight_dtype BF16 field
and unused config/run outputs, and never imports Torch or starts training.
Targets: physical8 x 4 ranks x accumulation1 = global32; LR G/F2e-6; 200G; beta
max.05/start20/ramp80; CD max.1/ramp100; F:G5; native1024/four-step G unchanged.
Activation checkpointing is proposed for batch8; fit/throughput must be measured
with beta/CD active and all anchors. Alternatives physical4/accum2/global32 and
physical4/accum1/global16 are explicit distinct run modes, not silent fallbacks.

This smaller batch and faster beta ramp are screening hypotheses, not paper
settings or guaranteed speedups. Start each science run from paired init with
fresh optimizers/counters. Preserve FP32 optimizer masters and source EMA updates
from FP32 master G; recipient must fix the reported BF16-online EMA source first.
Budget/control comparisons must declare actual samples, ramps/LR and GPU-hours;
final paper comparisons still require the full audited 10K evaluation protocol.
The legacy 2-GPU launcher/template is retained separately. No tests, model code,
GPU smoke, training or benchmarks were executed on the author's Mac for this update.
