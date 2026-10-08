# Experiment results

No experiments have been executed during initial authoring. No model or benchmark
has been produced. Fill a row only after saving the linked evidence on the
recipient machine.

| Run / commit / config hash | Init source | G or EMA | Protocol manifest | FID | CLIP-S and scale | ImageReward and scale | HPS v2.1 | HPS v3 | Added GPU-hours |
|---|---|---|---|---|---|---|---|---|---|

Record initial-checkpoint evaluation first. Keep pilot subset results separate
from full COCO10K evaluation. Include all metric versions/scales, prompt-reference
mapping, actual batch/world size, sampler/seed policy and repeatability evidence.
If a metric is absent, state why; do not substitute the reference paper's value.

For the H100 experiment, save initialization, screen candidates at 300 G, winner
and control at 1500 G, then both at 5000 total G for a matched final comparison.
Record physical batch, ranks, accumulation, effective batch (128), LR (5e-7),
CD maximum, beta ramp, diagnostic failures, wall/GPU hours and init→pilot→final
metric deltas. Screen scores do not establish full-beta effectiveness. Keep
validation selection separate from final benchmark; see H100_RUN_PLAN.md.
