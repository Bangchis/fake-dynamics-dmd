# Debugging and diagnostic interpretation

Use `fdmd --debug COMMAND ...` for full traceback. Read the earliest failure and
`metrics_rankN.jsonl` for every rank, not only the last NCCL timeout. Check the
source commit/config hash, current successful/attempted counters, mixed precision
and actual checkpoint before changing the method.

| Symptom | First things to check |
|---|---|
| strict import mismatch | Full paired release, raw generator state, exact `fake_unet.` prefix, UNet config and tensor shapes. Do not use `strict=False`. |
| no CUDA / wrong precision | Torch wheel, CUDA driver/device visibility, bf16 support, matching torchvision; doctor allocates no model. |
| out of memory | Check actual sharding/EMA/teacher settings, peak by rank, failed operation and batch. H100 mode shards optimizer state and stages CPU EMA; G/F and DDP buckets still occupy VRAM. Try a new run with physical 1/accumulation 64 to retain effective batch 128. FSDP is not implemented. |
| fake loss suddenly lower when beta rises | Target normalization divides by 1+beta. Inspect corrected errors and held-out probes, not only raw/mixed loss. |
| G has no gradient | Proxy y must retain graph; critic estimates and proxy target must detach. Detached teacher/fake MSE cannot update G. |
| teacher/EMA changes unexpectedly | Frozen `requires_grad`, eval mode, correct optimizer parameter identities; EMA after successful G only. |
| CD error rises / h_s out of range | Native scheduler, DDIM eta=0 formula, midpoint, raw F rather than corrected/CFG F, h_tau rather than critic x_t. Compare endpoint mean/std with baseline. |
| bright/clipped/broken images | VAE scaling factor, epsilon/x0 conversion and precision, no extra generator CFG, paired weights, native anchors/time IDs. |
| beta immediately max at fresh start | New k_G must be 0. Source checkpoint 019000 is provenance only. Check explicit beta override. |
| repeated skipped steps | JSONL has network, loss, norm, AMP scale and failure streak. The run aborts after the configured threshold; debug the first nonfinite value. |
| resume rejected | Scientific config, prompt hash, batch/world size and coefficients must match. Only budget/output/log intervals can change. |
| checkpoint checksum failure | Use a complete atomic directory; ignore `.incomplete_*`. Do not remove validation to load a partial write. |
| rank-local optimizer resume rejected | Both rank files, exact Torch build/world size and identical named-parameter partitions are required. A single common training.pt does not restore optimizers. |
| no TensorBoard scalars | Install logging extra in the training env; only rank 0 writes events. F uses k_F, G/probes use k_G. Null sparse buckets are intentionally omitted. |
| CPU EMA or functional_call failure | Verify Torch 2.6, device/dtype of the temporary state and FP32 CPU master. Snapshot is released before backward and invalidated after G/restore. Compare targets against the full FP32 reference on real SDXL. |
| evaluator import/model download failure | Clean pinned DMD2 checkout, eval dependencies, model cache/network, OpenAI CLIP installation if CLIP requested. |

No clipping/thresholding of latents or `nan_to_num` is added to hide instability.
Grad clipping is explicit at max norm 10 after one AMP unscale. Loss/gradient
nonfiniteness yields a logged skipped step and eventually a preserved failure
checkpoint. Successful-step counters and EMA do not advance on skips. A hardware
kill can leave the most recent in-flight cycle unsaved; resume a completed periodic
checkpoint instead of interpreting an incomplete directory as valid.

## What to log and interpret

F events: mixed epsilon MSE, actual beta, anchor, grad norm, successful flag,
corrected denoising error per timestep bucket and its sample count. These are
pre-update training-minibatch errors, not independent tests of lag.

G events: direct/CD/total losses, weights actually used, next-update schedule,
anchor/CD-active flag, G grad norm, branch output norms, weighted CD/direct y-space
gradient ratio and endpoint statistics. A y-space ratio is not a parameter-gradient
cosine or evidence of eliminated gradient conflict.

Fixed probes: independent deterministic RNG; raw-F output drift, full 4-step G
output drift with fixed re-noising seeds, CD residual, h_s/reference-input mean/std,
and held-out corrected errors on fresh generated four-step endpoints. Per-rank
probe history resumes with checkpoints. Each bucket can have few/no observations;
increase sampling before making comparisons. Probes add real F/G/T/EMA forwards.

Wall time includes prior training-loop work/periodic checkpoints after initialization;
the final checkpoint has its own logged write_seconds and is outside the saved
elapsed-training counter. Allocated
GPU hours are world-size × wall-hours on CUDA. Forward counts are logical calls
and exclude activation recomputation. Measure real utilization/throughput/memory
with your profiler; no measurement is supplied in this repo.

## Recoverable-underfitting probe for critic lag research

At a selected checkpoint, make an isolated diagnostic copy of F and freeze G.
Keep beta fixed. Generate held-out clean/noisy samples with the **random-anchor
training mixture**, and fit the F copy on independent generated/noisy samples.
Evaluate corrected epsilon error on the same held-out draws before/after extra
F-only updates and save the error curve. Never overwrite the actual training F
or reuse fit minibatches as test data. This optional experiment is a recipient
extension; no automatic lag-fitting controller is included in v0.

A large improvement is evidence of recoverable fitting deficit, not exact oracle
score error. Compare tracking, teacher bias and EMA staleness separately. Fixed
beta's mix/correction has the same linear tracker contraction as ordinary critic
tracking; any benefit must arise from learned optimization/generalization or less
abrupt G drift. Slow beta ramp/EMA alone proves none of those claims.
