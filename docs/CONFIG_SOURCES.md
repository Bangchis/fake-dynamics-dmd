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
| G/F learning rate | paper + script: 5e-7 | **1e-6**, owner-selected (2× reference); constant, fresh optimizers |
| AdamW betas / decay | paper | (0.9, 0.999) / 0.01 |
| Gradient clipping | script | 10 |
| Fake updates per G | paper + script | 5 successful F : 1 successful G |
| Teacher CFG | paper + script | 8 |
| Generator anchors | paper + script | 999, 749, 499, 249; stochastic re-noising |
| Resolution | script | 1024 |
| Global batch | paper: 128 (64 GPUs × physical 2) | **128**: physical 2 × 2 ranks × accumulation 32 |

Physical batch 2 has not been measured on these H100s. If it does not fit, physical
1 × 2 ranks × accumulation 64 preserves effective batch 128 and LR 1e-6. Matching
batch alone does not establish a paper reproduction: accumulation is our adapter,
and LR is deliberately doubled at the owner's request. The published setting includes GAN supervision and real
images; this continuation uses prompt-only objectives. Initialization retains its
DMD2 GAN history. The source checkpoint label 019000 is upstream outer-loop
provenance; this repo's counter always measures actual successful G updates.

### Teacher CFG and generator sampling

Teacher CFG **8** is explicit in Appendix F.4 and `real_guidance_scale=8` in the
pinned script. `fake_guidance_scale=1.0` refers to the critic's conditional output.
It must not be mistaken for an extra guidance coefficient applied to the generator.
The [official full-weight SDXL demo's sample method](https://github.com/tianweiy/DMD2/blob/8d8fa55633d47cfb81bbc7a892e7248f9518763f/demo/text_to_image_sdxl.py#L142-L176)
calls G once with prompt embeddings at each of 999/749/499/249, converts epsilon
to x0, then adds fresh noise. There is no unconditional G pass or CFG combination.
This repo follows that sampler. The inference convention is conditional-only
(`guidance_scale=1` in conventional CFG notation), while teacher CFG=8 is distilled
through the training objectives. Applying ordinary SDXL external CFG afterward
changes this trained sampler; it is not the baseline evaluation configuration.

### Accumulation and comparison policy

G/F parameters stay fixed throughout each optimizer window. One anchor is shared
across ranks and all microbatches, preserving DMD2's shared-minibatch anchor rule.
DM/CA/noise remain independently drawn per sample. Each mean microbatch loss is
divided by accumulation; DDP averages ranks, synchronizing on the final backward.
Unscale/clip/AdamW/EMA run once. Beta/CD schedules and F:G ratio count successful
optimizer windows. See [PyTorch 2.6 DDP no_sync](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/parallel/distributed.py#L1297-L1322).

Winner and beta=0/CD=0 control must use the same LR, physical/effective batch,
initial G/F hashes, prompt order/seed, G/F update budget, resolution, sampling and
metric protocol. Compare at 1500 G; for a claimed final-method gain at 5000 G,
continue the control to 5000 too. Report processed samples, actual GPU-hours and
the inherited DMD2 training separately. Extra teacher/CD calls make matched
update budgets different from matched wall-time budgets; report both costs.
Published DMD2/Decoupled scores remain external references until evaluation parity
is established. Do not infer their successful-G budget from an outer-loop label.

`python -m fake_dynamics.comparison RUN_A RUN_B --allow-field FIELD` audits actual
v0.3 checkpoint JSON metadata without loading models. Declare only intended
research knobs; it flags additional changes, incomplete/mismatched update budgets,
initialization/data/code differences and skipped windows. Optional paired
evaluation reports audit sample/seed/weights/reference/metric protocols as well.
It reports GPU-hours separately and does not certify significance or paper parity.

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
