# Verification status and recipient checklist

Initial delivery date: 2026-10-08. The owner's final constraint was not to run tests
on the author's small Mac CPU. **No unit tests, toy smoke, SDXL smoke, real-weight
import, CUDA/DDP/AMP run, training, save/resume execution or metrics were run.**

Authoring checks are limited to code review and lightweight static syntax/lint/
format analysis. A clean static check is not evidence of runnable SDXL or correct
training. Runtime dependencies and tests are authored for the recipient.

Static authoring results: Ruff lint and formatting passed; Python AST parsed the
19 Python files without importing them; Bash syntax checks passed for both shell
scripts. These checks do not execute model code or the authored tests.

The GitHub workflow is **workflow_dispatch only**. Publishing this repo does not
automatically run the tests; the recipient may trigger it explicitly.

## Authored tests, all pending execution

| File | Intended meaningful coverage | Initial status |
|---|---|---|
| `tests/test_algebra.py` | mixed target, epsilon/x0 correction, same-state CA+DM guided identity, proxy sign/reduction, coefficient broadcasting, inclusive schedules, G-counter ramps | not run |
| `tests/test_sampler.py` | stochastic G anchors, on-policy inputs, two-step raw fake DDIM, analytic constant-epsilon transition, no 249→0 CD | not run |
| `tests/test_training.py` | gradient ownership, EMA timing/dtype, skipped step counters, prompt-only stream, strict F extraction, atomic save/resume/RNG/next-cycle replay, midcycle slots, scientific config mismatch | not run |
| `tests/test_config.py` | unknown fields fail, unsupported flags fail, runtime machine budget remains unset | not run |

Run on the recipient machine from repo root after installing dev dependencies:

```bash
ruff check .
ruff format --check .
pytest -q
fdmd --debug train --config configs/toy.yaml
```

For CI, open Actions → Recipient verification (manual) → Run workflow. Save its
URL/commit/results with the handoff. CPU/toy passing still leaves SDXL integration
and benchmark unverified.

## Before a long SDXL run

- [ ] Record hardware/CUDA/driver/Python/packages, full host memory/disk availability.
- [ ] Verify downloaded asset revisions/hashes and actual G/F keys/tensor shapes.
- [ ] Confirm teacher scheduler/prediction type and conversion tolerance.
- [ ] Run numerical/ownership/sampler tests; retain outputs with commit/config.
- [ ] Confirm two text encoders, pooled/time-ID/unconditional conventions in real forwards.
- [ ] Run GPU smoke with nonzero beta/CD; see each required CD anchor and 249 skip.
- [ ] Confirm only F or G changes in its step, T frozen/eval, EMA changes after G only.
- [ ] Exercise actual AMP skip detection without advancing G/EMA and bounded failure handling.
- [ ] Save/resume real model, optimizer/scaler/counters/RNG/order/schedule with consistent next update.
- [ ] If using DDP, check synchronized anchors/skips, disjoint prompt slices and per-rank restore.
- [ ] Measure VRAM/host RAM/checkpoint I/O/throughput; set actual batch and run budget.
- [ ] Evaluate initial checkpoint; audit sampler/seed/metric protocol and reference mapping.
- [ ] Pilot stability/fixed samples/probes; distinguish sparse/null bucket diagnostics.
- [ ] Run beyond the beta ramp, select checkpoints without using eval training data.
- [ ] Full 10K evaluation of a fixed G or EMA choice, with all metric provenance.

Change checkbox status only when actual evidence is available. No training or
benchmark success can be deduced from the fact that source code was published.
