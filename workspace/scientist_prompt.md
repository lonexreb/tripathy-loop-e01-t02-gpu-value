# AI Scientist research playbook (mutable; rewritten by the agent each iteration)

## Mission
Beat the champion on the four Holdout gates (pooled skill, breadth, calibrated 80% band, no bad horizon)
with forecasts that would also be honest at the frozen 2026-11-06 origin.

## Where the loss lives (check before every hypothesis)
- The pooled score is scale-dependent and *fragile*: one exploding cell can annihilate it (iteration 2: a
  single series with width_rel 1e86 drove pooled skill to -1e84). Read `slices[].share_of_total_loss`
  and `worst_series` first; ~95% of champion loss is AWS hyperscaler_spot (h7 60%, h3 25%, h1 9%).
- Fix the regime that carries the loss. A gain on a 1% slice cannot move the pooled skill.
- **Read `mean_q50_error` vs `mean_width_rel` in worst_series before blaming the point forecast.**
  Iteration 4 showed the worst calm-zone series (p5 use1 az1/az4, skill -0.99) have q50 error of only
  -0.006 with coverage 0.97: their loss is WIDTH (over-coverage), not q50 bias. Diagnose the error
  channel (bias vs width) before choosing a lever.

## Method
1. Diagnose from Discovery only: per-slice skill, q10/q90 breach asymmetry, worst series (bias vs width).
2. State ONE falsifiable hypothesis with a mechanism (why prices behave that way) and a predicted sign.
3. Change one thing at a time so the ledger attributes the effect; keep everything else identical.
4. Record the result in the notebook, including negative results and their magnitude.

## HARD guardrails (learned the hard way — iterations 2, 3 AND 4 reverted)
- **Bound every output.** Clip per-series z to a small multiple of pooled z AND clip final log-quantiles to
  a sane band around the last value (log q within +/- 1 of log last). Iteration 2's explosion (h7 pooled
  -1e18) came from unbounded per-series z on a sticky series.
- **Never touch the LOWER quantile with a "max".** Iteration 3 falsified "elementwise max of pooled z is
  widening-only": max widens the upper side but RAISES q10 (less-negative z10), narrowing the lower band.
  Result: q10 breach 10.1% -> 10.9%, holdout coverage 0.7408 -> 0.7380, Gate 3 fail. Any upper-tail
  widening experiment must leave z10 byte-identical to the champion.
- **Recent-window residuals do NOT estimate forward tails (falsified twice).** Iteration 2: per-series
  recent-window z narrowed bands and exploded. Iteration 3: pooled split-conformal recent-90-day z,
  even taken as max with the full-window z, moved q90 breach only 11.9% -> 11.6% and gained just +0.0004
  pooled. The champion's full-window in-sample pooled z is already honest on the upper side.
- **Validation-selected parameter changes that can also select "no change" tend to degenerate to the
  champion on holdout** (iteration 4: shrinkage s=1.0 chosen at every holdout origin -> holdout aggregates
  byte-identical -> Gate 1 fails by construction). If a lever needs a *gain*, it must change something
  unconditionally or its selection rule must be shown (on Discovery) to actually fire.
- **Holdout coverage 0.7408 is 0.0008 above the Gate-3 floor.** Any change that lowers coverage even
  slightly fails Gate 3. Coverage-neutral-or-widening changes only; never narrow.
- Point-in-time only: use rows dated on or before the origin; never hard-code dates or per-origin constants.
- Pool parameters across series where per-series histories are short (< 60 days).
- Code: numpy/pandas only, vectorised, < 20 s per origin; guard empty histories and NaNs; always return
  every target series; enforce q10 <= q50 <= q90 and positivity.

## Current champion state (iteration 1, promoted; iterations 2, 3, 4 reverted against it)
Discovery pooled 0.4429, median 0.2977, coverage 0.7795, breaches 10.1%/11.9% (upper tail slightly
under-covered). Holdout pooled 0.2966, median 0.2552, coverage 0.7408, horizons 0.398/0.324/0.267.
Regimes: A AWS spot = pooled momentum OLS on 1/3/7-day log changes + local-vol conformal (full-window
in-sample pooled z); B rate cards = empirical change quantiles + stickiness pin + IQR width; C
marketplace = drift, cross-collector pull disabled (kappa = 0, falsified in iteration 1).

## What the levers are worth (all measured, all small or dead)
- Upper-tail widening via max(full-window, recent split-conformal) z: +0.0004 pooled, -0.0028 coverage.
  Dead lever — do not retry recent-window conformal in any form.
- Per-series local calibration: catastrophic (iteration 2). Dead lever.
- **Momentum-beta shrinkage toward zero, validation-selected (iteration 4): DEAD.** Holdout aggregates
  byte-identical to champion (validation always picked s = 1.0); where it fired on Discovery h1 it
  slightly hurt (h1 0.4294 vs 0.4300, pooled 0.44279 vs 0.44285). The pooled OLS beta is already at the
  out-of-sample optimum — the "calm zones mean-revert so shrunk betas help q50" mechanism is FALSIFIED.
  The calm-zone worst series lose on WIDTH (coverage 0.97), not q50 bias.
- Cross-collector pull (kappa > 0) and cross-AZ consensus: falsified in iteration 1. Dead levers.

## Structural reading after four iterations
The champion appears near a local optimum for PACK_ONLY signals: every calibration lever is falsified,
and the one point-forecast lever (beta shrinkage) degenerates to no-op. The remaining diagnosed
asymmetries are: (i) q90 breach 11.9% vs q10 10.1% (upper tail slightly under-covered), (ii) h1
coverage 0.840 too wide vs h7 0.769 too narrow (Discovery), (iii) calm-zone over-coverage (0.97) which
is a width loss but narrowing it risks the holdout Gate-3 floor.

## Next candidate ideas (ranked, expected values honest)
1. **Asymmetric multiplicative z90 inflation only** (z10 and z50 byte-identical to champion): scale the
   regime-A pooled z90 by a fixed factor ~1.10 chosen point-in-time from the trailing full-window breach
   excess (q90 breach 11.9% vs 10% target). Widening-only, coverage-safe (only raises coverage), cheap.
   Iteration 3's evidence caps the expectation at ~+0.001 pooled; it is the only lever with a positive
   measured sign. Falsified if holdout pooled <= 0.2966 + 0.003 (i.e. if the widening cost on the ~88%
   inside cells exceeds the 0.9-weighted breach saving).
2. **Per-horizon width reallocation in regime A** (narrow h1 z, widen h7 z, pooled coverage-neutral by
   construction): h1 coverage 0.840 vs h7 0.769 suggests h1 bands waste width. DANGEROUS: holdout
   coverage sits 0.0008 above the floor, so any net narrowing fails Gate 3. Only attempt with an
   explicit constraint that aggregate holdout coverage cannot fall (e.g. widen h7 first, narrow h1 by
   less than the h7 gain), and expect it to be a no-op or a Gate-3 fail.
3. **Accept the champion.** After three falsified calibration levers and one falsified point-forecast
   lever, the honest prior is that PACK_ONLY headroom above 0.2966 pooled is < 0.005. If idea 1 fails,
   recommend freezing the champion rather than risking reverts.

## Notebook (append-only ledger of what was tried)
- Iteration 0 (SEED): naive baseline, holdout pooled 0.2306.
- Iteration 1 (PROMOTED): multi-regime momentum + local-vol conformal. Holdout pooled 0.2966,
  coverage 0.7408. This is the champion.
- Iteration 2 (REVERTED): per-series recent-window split-conformal z. Exploded (h7 pooled -1e18,
  unbounded per-series z on a sticky series). Lesson: bound every output; per-series calibration is dead.
- Iteration 3 (REVERTED): pooled recent-90-day split-conformal z taken as max with full-window z.
  +0.0004 pooled, coverage 0.7408 -> 0.7380, Gate 3 fail. Lesson: recent residuals are not
  heavier-tailed than full-window; upper-tail widening via recent windows is dead.
- Iteration 4 (REVERTED, clean null): regime-A momentum-beta shrinkage toward zero, s in [0.5, 1.0]
  chosen by pooled out-of-sample validation on the recent 90 days (beta refit on the older chunk,
  >1% MSE-gain threshold). Holdout aggregates BYTE-IDENTICAL to champion (validation picked s = 1.0
  at every holdout origin); on Discovery it fired only at h1 and slightly hurt (pooled 0.44279 vs
  0.44285, h1 0.4294 vs 0.4300, q10 breach 10.05% vs 10.11%). FALSIFIED: the pooled OLS momentum beta
  is already out-of-sample optimal; the calm-zone worst series (p5 use1 az1/az4, skill -0.99) lose on
  band WIDTH (coverage 0.97, width_rel 0.059, q50 error only -0.006), not on q50 bias. Lesson: diagnose
  bias-vs-width before touching the point forecast; validation-gated changes that can select "no
  change" degenerate to the champion and cannot pass Gate 1.
