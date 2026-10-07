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
| FP32 train/EMA state; optional network autocast | Prevents small EMA updates rounding away. Full memory is substantial; no claim of fitting small GPUs. |
| Single/DDP only, accumulation 1 | Refuse unsupported memory flags rather than silently pretending FSDP/accumulation is correct. Both modes await actual execution. |
| Teacher fallback pauses after fake warmup | Fixed number alone does not prove fitting adequacy; recipient inspects diagnostics before continuing. Paired F remains the default. |
| Nonfinite gradients are exposed | No latent clipping/thresholding/nan_to_num masking. Log AMP skips, preserve failure checkpoint, abort repeated failures. |
| Atomic retained checkpoints | Never delete initial/best candidates. Exact resume requires same prompt contents, batch and world size. |
| Per-prompt independent sample seed | Clear stable mapping/re-noising sequence. Differs from original batched evaluator RNG and is disclosed as a protocol discrepancy. |
| Pinned upstream FID/CLIP bridge | Reuses inspected metric functions. External clean-fid components/feature caches still require calibration; other metric versions/scales use explicit plugins. |
| No testing in the authoring session | Owner explicitly stated the Mac CPU is small and requested no tests. Only lightweight static syntax/lint/format checks are permitted here; all execution evidence awaits recipient. |

Future scientific decisions should record trigger/evidence, changed config/code,
expected behavior, verification status and benchmark implications. Do not quietly
add GAN, boundary CD at 0, shared weights, teacher trajectory CD, EMA-F or an
adaptive controller. Diagnostic ablations use explicit config overrides and a
new run, preserving the full method's provenance.
