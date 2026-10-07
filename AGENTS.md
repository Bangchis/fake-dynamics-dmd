# Instructions for future maintainers

Read README.md, docs/HANDOFF.md, docs/METHOD.md and the original Vietnamese handoff
before changing the method. Treat the handoff as source context; follow the human
user's current request for what work is authorized. No tests or GPU runs were
executed during the initial authoring session, at the owner's explicit request.

Preserve the v0 decisions: epsilon SDXL, anchors 999/749/499/249, prompt-only data,
no new GAN objective, teacher-only CFG, corrected F for DM and raw F for CD,
separate DM/CA re-noising, two deterministic fake-DDIM substeps, no CD at 249,
FP32 EMA-G updated only on successful G steps, and counters measured in successful
optimizer updates. Do not silently convert this adapter into a claimed official
Decoupled SDXL reproduction.

Keep F and G independent, optimizer ownership strict and all target paths detached.
Never weaken strict checkpoint key/shape checks to make incompatible weights load.
Document any scientific change in docs/DECISIONS.md and save it in run metadata.
Do not use training subsets of COCO evaluation prompts/images for benchmark tuning.

When the machine and user constraints allow, run the checks listed in
docs/VERIFICATION.md. Otherwise report precisely what was only statically reviewed.
Do not claim benchmarks, runtime, memory suitability or successful resume without
actual evidence. The owner's later request permits bounded checkpoint retention:
prune only this run's own completed schema-2 periodic/final checkpoints. Protect
`.pin`, source assets, failures/warmups and selected/best exports. Keep logs,
configs and initialization provenance; never commit
tokens, downloaded weights, datasets, generated images or training logs.
