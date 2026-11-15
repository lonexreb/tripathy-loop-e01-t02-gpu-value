# ADR 0002: Supersede the Kalshi scope with Track 2 Direction A

- Status: accepted
- Date: 2026-11-14

## Decision

Tripathy Loop E01 will forecast q10/q50/q90 posted GPU rental prices in USD per GPU-hour at 1, 3, and 7-day horizons, using mean pinball loss and the official `last_value`/`drift` baselines from `gpu-price-pack-v0.1.1-handover`.

The provisional Kalshi replay, trading simulation, and Brier-score framing is superseded. The former documents are preserved under `docs/superseded/kalshi/`, and commits `ac039bf` through `8189e37` remain unchanged in Git history.

## Reason

The complete five-page Pacific Track 2 brief and its supplied data pack define a continuous quantile-forecasting task. Kalshi is not the scored object, and treating it as the core benchmark would misstate the contract.

## Consequences

- The active baseline is the starter `drift` forecast; `last_value` is the skill denominator.
- The frozen evaluator enforces origin cutoffs and preserves negative results.
- Instrument classes remain separate; no synthetic cross-instrument price curve is allowed.
- Recursive-self-improvement claims require the common-start test in `research/PROTOCOL.md` and cannot be inferred from one improved forecast alone.
