# Project brief — Track 2 GPU Value

## Objective

Build a research system that proposes a falsifiable change to a GPU rental-price forecaster, evaluates it under a frozen time-causal protocol, keeps only changes that survive unseen-origin tests, records positive and negative results, and repeats.

## Forecast contract

- Target: posted rental-price series in USD per GPU-hour, never a synthetic splice of instruments.
- Origin: 2026-11-06.
- Horizons: 1, 3, and 7 days.
- Forecasts: q10, q50, q90.
- Metric: mean pinball loss; skill denominator `last_value`; practical baseline `drift`.
- Calibration target: 80% of truths inside [q10, q90].

## Current scope

The adapted research slice contains seven H100/A100 headline series with daily truth. Other final-origin headline cells use the disclosed drift fallback. AWS spot and horizon 30 remain development or ledger legs because Sunday truth is unavailable under the pack contract.

## Scientific claim

Cycle 1 supports a forecasting improvement claim: trailing conformal intervals improved checkpoint pinball loss and coverage over drift. It does not yet support a recursive-self-improvement claim; that requires the Cycle 2 common-start comparison specified in `research/PROTOCOL.md`.

## Superseded scope

The former Kalshi/Brier-score project framing is preserved under `docs/superseded/kalshi/` and superseded by ADR 0002. Its commits remain part of the repository history.
