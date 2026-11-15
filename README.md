# Tripathy Loop E01 — Track 02: GPU Value

This repository implements Pacific Track 2 Direction A: a leakage-safe research system that proposes bounded changes to a GPU rental-price forecaster, evaluates them on unseen historical origins, preserves negative results, and advances only candidates that pass the frozen acceptance rule.

The scored object is a set of `q10`, `q50`, and `q90` forecasts in USD per GPU-hour at horizons 1, 3, and 7 days. The primary metric is mean pinball loss; headline skill is measured against `last_value`, and `drift` is the practical baseline to beat.

## Current checkpoint

- Data pack `v0.1.1-handover`: published ZIP checksum verified; 672/672 manifest entries verified.
- Frozen protocol: seven adapted H100/A100 posted-rental series, with disclosed drift fallback for the remaining final-origin headline series.
- `F_0`: official drift baseline.
- Cycle 1 `I_0`: two candidates proposed within budget.
- `repricing_mixture`: rejected because its checkpoint interval crossed zero.
- `conformal_trailing(window=45)`: accepted as `F_1`; checkpoint pooled skill improved from `+0.0891` to `+0.1186`, and interval coverage improved from `0.6655` to `0.7138`.

## Reproduce

The immutable pack, virtual environment, and generated outputs remain local and are ignored by Git.

```bash
bash research/reproduce.sh
```

The command verifies the pack, runs unit checks, rebuilds the baseline and Cycle 1, validates the submission, performs the expected leakage refusal, and prints deterministic hashes.

## Repository map

- `PLAN.md`: governing contract and execution plan.
- `research/PROTOCOL.md`: frozen experimental protocol.
- `research/origins.json`: development, checkpoint, protected, and official blocks.
- `research/scientist/`: candidate forecasters, improver, and evaluator.
- `research/lineage/`: accepted and rejected machine-readable lineage.
- `docs/adr/0002-supersede-kalshi-scope.md`: scope migration decision.
- `docs/superseded/kalshi/`: preserved former scope documents.

## Lane

E01 continues to use the existing GitHub history, Linear project, and Slack channel. Git history is preserved; the earlier Kalshi framing is superseded, not rewritten.

## Submission

- Findings: [FINDINGS.md](FINDINGS.md)
- Forecasts: `submission/submission_head_2026-11-06.csv` (SHA-256 in `submission/SHA256SUMS`)
- One command to rebuild them from the pack: `bash research/reproduce.sh`
