# Method and implementation map

The research hypothesis is that a teacher-anchored fake field supplies consistency
targets that are better matched to a changing generator. It has not been shown to
improve stability, critic lag, variance, convergence or FID. The mix/correction
identity alone gives no free statistical benefit. Read original_handoff_vi.md for
motivation, prior art and mathematical limitations.

## Invariants

Native DDPM indices decrease toward clean. `alpha=sqrt(alphas_cumprod[t])`,
`sigma=sqrt(1-alphas_cumprod[t])`, `D_t(x,e)=(x-sigma*e)/alpha`. All coefficients
come from the checkpoint scheduler; batch coefficients reshape to `[B,1,1,1]`.
Conversion defaults to float64 as in the inspected DMD2 training helper; network
outputs/losses/renoised states are FP32 with optional network autocast.

G/F are separate full SDXL UNets. T, two text encoders and EMA-G are frozen. T/EMA
are in eval mode. VAE is deferred until decode because training uses only latents.
Conditioning uses both penultimate text hidden states, encoder-2 pooled output and
time IDs `[height,width,0,0,height,width]`. T unconditional embeddings are zero;
time IDs remain identical. CFG is applied only to T.

For a shared minibatch anchor tau, on-policy G backsimulation makes h_tau, and the
online G predicts clean y. Fake/DM/CA re-noise **y**, while CD starts from **h_tau**.
G uses anchors `[999,749,499,249]` and fresh stochastic re-noising. Fake DDIM is
restricted to the auxiliary CD path; it does not replace G inference.

## Losses

Fake: `target=(noise + beta*T_CFG)/(1+beta)`; mean epsilon MSE. No min-SNR or x0
regression weighting. G/T/target detach. At beta=0 the teacher fake-fit call is
skipped and target is the sampled noise.

DM correction: `qhat=(1+beta)*F-beta*T_CFG` at the **same** noisy state/time/c.
The implementation corrects x0 estimates with the equivalent affine expression.
Raw F is retained for CD. The ideal correction amplifies fake fitting error by
`1+beta`; it does not ensure critic lag falls.

Direct branch gradients:

```
N_a = max(mean_per_sample(abs(y - x0_T_CFG(a))), 1e-6)
g_DM = (x0_qhat(t_DM) - x0_T_cond(t_DM)) / N_DM
g_CA = (w-1)*(x0_T_uncond(t_CA) - x0_T_cond(t_CA)) / N_CA
g = g_DM + g_CA
L_direct = 0.5 * mean((y - stopgrad(y - g))**2)
```

Optimizer descent moves y along `-g/numel(y)`. Critic outputs and normalizers
detach; y keeps its G graph. `t_F` is inclusive 0..999, `t_DM` 20..980, `t_CA`
20..min(980,tau-1), with independent DM/CA noise. At beta=0 and identical noisy
state/time, DM+CA reduces to the DMD2 guided numerator/normalizer. With separate
schedules this is the declared no-GAN Decoupled adapter, not a proven exact
reproduction of the official SDXL Decoupled implementation.

CD pairs: 999→749, 749→499, 499→249. Midpoint is floor((tau+s)/2). Raw conditional
F gives two deterministic eta=0 DDIM transitions. Endpoint target is frozen EMA-G
x0 at s. Loss is mean latent x0 MSE; tau=249 skips CD. No 249→0 boundary.

`L_G=L_direct+lambda_CD*L_CD`. EMA-G is initialized from the loaded G, remains
FP32 and updates once per successful G optimizer step with decay 0.99. No EMA-F,
weight merging, adaptive controller, GAN loss or generator external CFG.

## Time scales

New run: k_G=k_F=0, fresh AdamW. Source checkpoint label 019000 is only provenance,
not the new schedule start. Five successful F updates precede each successful G
update. Skipped F steps retry the current slot; skipped G steps retry G without
another five F steps. Finite gradients/AMP scale, not AdamW return value, determine
success. Repeated nonfinite updates produce a failure checkpoint and abort.

`lambda_CD=0.1*clip(k_G/100,0,1)`.
`beta=0.05*clip((k_G-100)/1000,0,1)`.
Beta stays fixed during all F slots and the following G step. The log records the
weights actually used; successful G events also give the next-update schedule.
The LR is constant per network in v0; no inherited outer-iteration scheduler.

Teacher-fallback F uses an explicit fake-only warmup with beta=0. The trainer then
stops with diagnostics/checkpoint; a recipient who reviewed fitting can resume
with `--continue-after-warmup`. Warmup counts do not advance k_G or its ramps.

## Requirement-to-code traceability

| Source requirement | Implementation | Recipient check |
|---|---|---|
| §§3,5 coefficient shape/precision | `diffusion.py` | batched coefficients; scheduler parity |
| §4 mixed target + corrected estimate | `objectives.py`, `trainer.fake_step` | beta=0; affine epsilon/x0 identity |
| §5 CA/DM signs/normalizers/schedules | `objectives.direct_gradient`, `diffusion.ca_time` | same-input guided identity, descent sign |
| §6 raw fake two-substep CD | `sampling.fake_transition`, `trainer.generator_step` | pairs, midpoint, detach, skip at 249 |
| §§6–7 EMA and G-unit ramps | `trainer.update_ema`, `config.weights` | no F-step EMA; overflow does not advance G |
| §8 paired full weight import | `weights.py`, `backend.py` | actual strict keys/shapes and provenance |
| §8 prompt-only no-GAN path | `data.py`, `backend.encode`, `trainer.py` | no LMDB/images/GAN accesses |
| §§8,10 success counters/5:1 | `trainer.optimizer_step`, `trainer.run` | nominal ratio, retries, new counter 0 |
| §11 save/resume | `checkpoint.py`, `runtime.py`, `PromptStream` | same next update/RNG/prompt order, rank topology |
| §12 diagnostics | `trainer.probe`, JSONL metrics | fixed drift, held-out errors, CD endpoint statistics |
| §13 sampler + evaluation provenance | `inference.py`, `evaluation.py` | checksum mapping, protocol equivalence audit |

Fixed probes measure raw-F output drift and G full-sampler output drift, CD residual,
endpoint mean/std, and held-out denoising errors on four-step G endpoints. The
last statistic is a practical probe, not the full random-anchor training-mixture
tracking error. Sparse timestep buckets can be null; record their sample counts.
G events report CD/direct gradient ratios in **output y space**, not parameter
gradient cosine/alignment. Logical forward counters include probes and exclude
extra recomputations from activation checkpointing; elapsed/allocated GPU hours
are wall-time estimates, not a profiler or measured utilization.
