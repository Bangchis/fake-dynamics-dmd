# Config rationale and original sources (reviewed 2026-10-08)

The source documents inform the implementation; the owner's current request sets
scope. `configs/h100_2x80.yaml` is an exploratory continuation experiment from
paired DMD2 weights. No setting has been tuned on the recipient's H100s yet.

## DMD2: closest full-weight SDXL reference

[DMD2 §4.2/4.4 and Appendix F.4](https://arxiv.org/html/2405.14867v2#A6.SS4)
and its [pinned SDXL launch script](https://github.com/tianweiy/DMD2/blob/8d8fa55633d47cfb81bbc7a892e7248f9518763f/experiments/sdxl/sdxl_cond999_8node_lr5e-7_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch.sh)
provide the primary optimizer/sampler reference:

| Setting | Reference | Selected config |
|---|---|---|
| G/F learning rate | paper + script | 5e-7, constant, fresh optimizers |
| AdamW betas / decay | paper | (0.9, 0.999) / 0.01 |
| Gradient clipping | script | 10 |
| Fake updates per G | paper + script | 5 successful F : 1 successful G |
| Teacher CFG | paper + script | 8 |
| Generator anchors | paper + script | 999, 749, 499, 249; stochastic re-noising |
| Resolution | script | 1024 |
| Global batch | paper: 128 | 2 initially; hardware adaptation, accumulation 1 |

Do not scale LR mechanically with the 64-fold batch reduction. Measure fitting and
update variance first. The published setting includes GAN supervision and real
images; this continuation uses prompt-only objectives. Initialization retains its
DMD2 GAN history. The source checkpoint label 019000 is upstream outer-loop
provenance; this repo's counter always measures actual successful G updates.

## Decoupled DMD: schedule rationale

[Decoupled DMD §4.3 / Table 2](https://arxiv.org/html/2511.22677v1#S4.SS3)
separates CA and DM re-noising. Its SDXL comparison keeps DMD2's configuration,
including GAN. We adopt the handoff's explicit native-DDPM adapter: independent
DM/CA draws; DM 20..980; CA 20..min(980, anchor-1). Keep this difference and the
branchwise CFG-reference normalization visible when comparing results. The
reported FID threshold is a target, not a measurement by this repo.

## Nearby consistency methods: principles, not interchangeable weights

[Consistency Models, Algorithm 2 and Table 3](https://proceedings.mlr.press/v202/song23a/song23a.pdf)
support detached target networks and distinguish target EMA from evaluation EMA.
Their backbones, solvers, distances and batching differ. Target decay 0.99 appears
in a pretrained consistency-training ablation; it does not establish the optimum
for raw-F trajectories in SDXL. We keep 0.99 as the handoff hypothesis (roughly
100 G updates of memory), with a FP32 master and successful-G-only updates.

[Flash Diffusion, Experimental Details / Flash SDXL](https://arxiv.org/html/2406.02347v3)
uses a LoRA student, LPIPS, adversarial supervision and staged objectives. Its 1e-5
learning rate and loss coefficients are not directly transferable to this
full-weight latent-MSE continuation. Staging motivates a bounded stability screen,
while our original ramps remain unchanged.

## Parameters that remain research hypotheses

| Parameter | Default | Origin / decision |
|---|---|---|
| Teacher anchor beta | max 0.05; starts after k_G=100; ramp 1000 G | original handoff proposal; not a published tuned value |
| CD coefficient | max 0.1; ramp 100 G | original handoff proposal; screen 0.03 / 0.1 / 0.3 |
| EMA-G target decay | 0.99 | handoff hypothesis; target/evaluation EMA currently share one state |
| Pilot / main budget | 1500 / 5000 total successful G | practical experiment cap; no prediction of convergence/FID |
| Frozen T / staged EMA forward | BF16 | memory adaptation; numerical parity awaits GPU checks |
| DDP + optimizer stage-1 sharing | 2 ranks | engineering choice; FP32 G/F/EMA master, no FSDP/LoRA |

The CD screen changes only CD maximum, with the same beta ramp, seeds, prompts,
physical batch and initialization. At 300 G, beta is only 0.01 for the next update;
this screen can reject unstable CD scales but cannot validate full-strength
teacher anchoring. Selected runs must pass 1100 G and reach 1500 before judging
the combined method. See H100_RUN_PLAN.md for selection and matched control.
