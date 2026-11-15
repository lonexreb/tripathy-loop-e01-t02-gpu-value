# F_0 baseline-only run — 2026-11-15

This run intentionally executed only the fixed baseline stage. It did not invoke `run_cycle1.py`, an improver, candidate generation, model selection, or any recursive self-improvement (RSI) logic.

## Command

```bash
./.venv/bin/python -I -B research/run_baselines.py
```

The command completed with exit code 0 and rebuilt its outputs from the unchanged `gpu-price-pack-v0.1.1-handover` data pack.

## Results

The adapted evaluation contains seven predeclared H100/A100 posted-rental series at horizons 1, 3, and 7 days.

| Block | Scored cells | `drift` skill vs `last_value` | `last_value_empirical` skill | `drift` 80% coverage | Paired pinball difference: `drift - last_value` | 90% CI |
|---|---:|---:|---:|---:|---:|---:|
| Development | 626 | +0.162595 | +0.170896 | 0.588 | -0.004581 | [-0.005881, -0.003372] |
| Checkpoint | 290 | +0.089106 | +0.092281 | 0.666 | -0.002892 | [-0.005056, -0.001043] |
| Protected | 255 | +0.196270 | +0.196975 | 0.663 | -0.008192 | [-0.011236, -0.005372] |

The negative paired differences mean that `drift` has lower pinball loss than `last_value`; all three confidence intervals exclude zero. However, `last_value_empirical` is slightly better than `drift` on every block, so the empirical median drift does not add measurable value here. Coverage remains below the 0.80 target on every block, making interval calibration the clearest weakness to address in later architecture work.

## Official-origin artifact

- Origin: `2026-11-06`
- Scope: 37 headline series
- Horizons: 1, 3, and 7 days
- Forecast rows: 111
- Output: `research/out/submission_F0_drift_2026-11-06.csv` (generated and intentionally ignored by Git)
- SHA-256: `50d49316c5b847ea781a583a69c5014413148800d604210ca373dc8b623b8dab`

The submission hash matches the earlier reproduction, confirming that this fixed baseline rerun is reproducible.
