# Attribution and scope

This research implementation follows the supplied fake-dynamics handoff and the
published SDXL training/sampling conventions of:

Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman,
Frédo Durand and William T. Freeman, **Improved Distribution Matching Distillation
for Fast Image Synthesis**, NeurIPS 2024.

Upstream: https://github.com/tianweiy/DMD2
Inspected commit: `8d8fa55633d47cfb81bbc7a892e7248f9518763f`.
Upstream license: Creative Commons Attribution-NonCommercial-ShareAlike 4.0.
The same license is provided in this repository for this implementation.

The code in `src/fake_dynamics/` is a new implementation: prompt-only data plumbing,
decoupled branch normalization, mixed fake targets, correction, auxiliary fake DDIM,
EMA, native optimizer/counter/checkpoint handling and protocol manifests. It does
not copy the full upstream trainer or claim endorsement by its authors. Evaluation
loads the recipient's pinned upstream checkout, which retains its own notices
including GigaGAN/BigGAN/clean-fid attribution.

The teacher/model checkpoints, captions and evaluation images are external assets
and remain subject to their respective model/data licenses. No weights or datasets
are redistributed here. See the supplied handoff for other research references and
for the distinction between inherited techniques and the hypothesis being tested.
