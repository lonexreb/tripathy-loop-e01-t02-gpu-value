# AI Scientist research playbook (mutable; rewritten by the agent each iteration)

## Mission
Beat the champion on the four Holdout gates (pooled skill, breadth, calibrated 80% band, no bad horizon)
with forecasts that would also be honest at the frozen 2026-11-06 origin.

## Where the loss lives (check before every hypothesis)
- The pooled score is scale-dependent; read `slices[].share_of_total_loss` and `worst_series` first.
- Fix the regime that carries the loss. A gain on a 1% slice cannot move the pooled skill.

## Method
1. Diagnose from Discovery only: per-slice skill, q10/q90 breach asymmetry, worst series.
2. State ONE falsifiable hypothesis with a mechanism (why prices behave that way) and a predicted sign.
3. Change one thing at a time so the ledger attributes the effect; keep everything else identical.
4. Record the result in the notebook, including negative results and their magnitude.

## Guardrails
- Point-in-time only: use rows dated on or before the origin; never hard-code dates or per-origin constants.
- Calibrate intervals from residuals of your own model (conformal-style), not from raw price changes.
- Keep the band inside 0.74..0.86 coverage: a sharper band that breaks Gate 3 is a failed experiment.
- Pool parameters across series where per-series histories are short (< 60 days).
- Code: numpy/pandas only, vectorised, < 20 s per origin; guard empty histories and NaNs; always return
  every target series; enforce q10 <= q50 <= q90 and positivity.
