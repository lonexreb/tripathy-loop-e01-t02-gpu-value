# AI Scientist research playbook (mutable; rewritten by the agent each iteration)

<!-- LAB-PROTECTED:BEGIN (owned by the lab; rewrites cannot remove or edit this block) -->
## Lab rules (protected)
- The lab, not you, decides when research stops. "Accept/freeze the champion" is never a valid plan: every
  iteration must test a real, falsifiable change. A string of reverts means the explored families are exhausted,
  not that the problem is solved; move to an untried family (see the experiment registry) or make a bolder change.
- Every hypothesis is tagged with one family and a novelty (incremental | bold). Do not spend more than two
  consecutive experiments in one family without a promotion; after two reverts in a row, the next one is bold.
- Near-misses are assets: reverted candidates are archived with their code, diff and diagnostics
  (inspect_candidate). Prefer building on the closest near-miss over restarting from the champion.
- A candidate whose Holdout aggregates equal the champion's (NO_OP) changed nothing that is scored: its
  selection rule never fired. Make changes that act unconditionally or show on Discovery that they fire.
- Guardrails are only removed with evidence: cite the iteration that falsified them in the rationale.
- Keep this playbook a distilled strategy (heuristics, guardrails, ranked ideas). The full experiment history
  is rendered by the lab from the ledger; do not copy the notebook into the playbook.
<!-- LAB-PROTECTED:END -->

## Mission
Beat the champion on the four Holdout gates (pooled skill, breadth, calibrated 80% band, no bad horizon)
with forecasts that would also be honest at the frozen 2026-11-06 origin.

## Where the loss lives (check before every hypothesis)
- The pooled score is scale-dependent and *fragile*: one exploding cell can annihilate it (iteration 2: a
  single series with width_rel 1e86 drove pooled skill to -1e84). Read `slices[].share_of_total_loss`
  and `worst_series` first; ~95% of champion loss is AWS hyperscaler_spot (h7 60%, h3 25%, h1 9%).
- Fix the regime that carries the loss. A gain on a 1% slice cannot move the pooled skill.
- **Read `mean_q50_error`/`bias_rel` vs `mean_width_rel` before choosing a lever.** Iterations 4-5
  established: h7 under-coverage (0.769) is a CENTERING problem (bias_rel -0.012 at h7, growing with
  horizon), not a width problem; the calm-zone worst series lose on WIDTH (coverage 0.97), not q50 bias.

## Method
1. Diagnose from Discovery only: per-slice skill, breach asymmetry, worst series (bias vs width).
2. State ONE falsifiable hypothesis with a mechanism and a predicted sign.
3. Change one thing at a time so the ledger attributes the effect; keep everything else identical.
4. Record the result, including negative results and their magnitude. Use analyze_discovery to measure
   out-of-sample BEFORE spending a dry run or a submission (budget: ~6 analyses, 4 dry runs per iteration).

## HARD guardrails (iterations 2, 3, 4, 5 — all measured, all falsified)
- **Bound every output.** Clip final log-quantiles to a sane band around the last value (log q within
  +/- 1 of log last). Iteration 2's explosion came from unbounded per-series z on a sticky series.
- **Never touch the LOWER quantile with a "max".** Iteration 3: max widens the upper side but RAISES q10,
  narrowing the lower band -> Gate 3 fail. Any upper-tail widening must leave z10 byte-identical.
- **Recent-window residuals do NOT estimate forward tails (falsified twice: iterations 2, 3).**
- **Validation-selected parameters that can select "no change" degenerate to the champion** (iteration 4:
  s=1.0 chosen at every holdout origin -> byte-identical aggregates -> Gate 1 fail by construction).
- **Holdout coverage 0.7408 sits at the Gate-3 edge.** Gate 3 passed at 0.738 in iteration 5 only via the
  "no worse than champion's distance" clause; treat any coverage drop as near-fatal. Coverage-neutral-or-widening.
- **In-sample mean residuals do NOT persist forward (iteration 5).** Adding the pooled per-horizon mean
  residual mu_h to all quantiles: pooled -0.0323, 11/11 origins worse. The forward bias_rel (-0.001/-0.006/
  -0.013) is NOT an intercept problem; do not "fix" bias with in-sample drift terms.
- **Per-series calibration is dead (iteration 2); per-type (instance_type) hierarchy is dead (iteration 5
  analysis):** per-type betas/z are a wash-to-worse out-of-sample (q50 err 0.01136 -> 0.01149 at h1).
  The pooled OLS beta is at the out-of-sample optimum.
- **Ensemble with a drift/persistence member is dead (iteration 5):** 0.8/0.2 momentum+drift quantile
  average: pooled -0.0141, 10/11 origins worse. The drift member degrades q50 and its wider tails cost
  more than the breach saving.
- **Calm-regime (bottom-quartile local vol) z narrowing is dead (iteration 5):** pooled -0.0005,
  coverage -0.004. The calm-zone over-coverage cannot be harvested by conditional narrowing either.
- **h7 tail widening is dead (iterations 3, 5):** +20% h7 tails: h7 skill -0.020 despite lifting h7
  coverage 0.769 -> 0.808. h7's loss is centering, not width.
- Point-in-time only: rows dated on or before the origin; no hard-coded dates or per-origin constants.
- Pool parameters across series where per-series histories are short (< 60 days).
- Code: numpy/pandas only, vectorised, < 20 s per origin; guard empty histories and NaNs; always return
  every target series; enforce q10 <= q50 <= q90 and positivity.

## Current champion state (iteration 1, promoted; 2, 3, 4, 5 reverted against it)
Discovery pooled 0.4429, coverage 0.7795, breaches 10.1%/11.9%. Holdout pooled 0.2966, median 0.2552,
coverage 0.7408, horizons 0.398/0.324/0.267. Regimes: A AWS spot = pooled momentum OLS on 1/3/7-day log
changes + local-vol conformal (full-window in-sample pooled z); B rate cards = empirical change quantiles
+ stickiness pin + IQR width; C marketplace = drift, kappa = 0.

## Confirmed-positive but sub-threshold lever (iteration 5, the closest near-miss)
**h1 tail narrowing (z -> z50 + 0.95*(z - z50) at h=1 only, q50/h3/h7 byte-identical): Discovery pooled
+0.0005 (10/11 origins better), Holdout pooled +0.00046, h1 skill +0.0047, gates 2/3/4 passed, Gate 1
shortfall 0.0025.** The mechanism (h1 band over-covers: 0.840 vs 0.80 target; the 0.1-weighted width cost
on inside cells exceeds the breach saving) is CONFIRMED on both Discovery and Holdout. It cannot pass
Gate 1 alone (+0.003 needed). Any future candidate should STACK it with a second independent positive
lever; do not re-submit it standalone.

## Structural reading after five iterations
Every width lever (widen/narrow, pooled/per-series, recent/full-window, per-horizon/per-regime) and every
centering lever (intercept, drift, shrinkage, hierarchy, ensemble) has been measured; all are negative
except the h1 narrowing (+0.0005). The champion is at a sharp local optimum for the signals it uses
(series_history only). Remaining headroom via re-calibration of the existing machinery is < 0.001.

## Next candidate ideas (ranked, honest expected values)
1. **exogenous_signals (UNTRIED, top priority): the pack tables the champion ignores.** Regime A uses only
   series_history. daily_statistics (p25/median/p75/min per collector), aggregator_medians (weekly),
   index_prints, hardware_fundamentals, capability_ladder/capability_units are all unused. Hypotheses:
   (a) cross-collector dispersion (p75-p25)/median of the SAME series predicts forward tail width better
   than the 14-day |1-day log change| vol (it is a contemporaneous demand-supply spread); (b) index_prints /
   hardware fundamentals give a level anchor for the h7 centering bias. Measure out-of-sample with
   analyze_discovery FIRST (the iteration-5 analysis protocol: train on d <= od-90, test on the recent 90
   days, pooled pinball vs champion machinery). Note iteration 1 falsified cross-COLLECTOR q50 pull
   (kappa) and cross-AZ consensus for regime C/A q50 — but NOT cross-collector WIDTH conditioning.
2. **Stack the confirmed h1 narrowing (+0.0005) with the best exogenous lever** if (1) measures >= +0.0025
   out-of-sample; the sum must clear +0.003 on holdout. Both levers must be shown to fire unconditionally.
3. **robustness (untried):** the champion has no log-band clip; add the iteration-4/5 guardrail (clip
   final log-quantiles to last log value +/- 1.0) as a pure safety change. Expected NO_OP on holdout
   (it never binds on sane series) -> Gate 1 fail by construction; only worth it bundled with (1)/(2).
4. **regime_change (partially burned):** calm-regime narrowing falsified; jump/step detection on AWS spot
   (sticky administered prices) is the remaining untested variant, but the prior after five falsified
   calibration levers is strongly negative.

## Notebook (append-only ledger of what was tried)
- Iteration 0 (SEED): naive baseline, holdout pooled 0.2306.
- Iteration 1 (PROMOTED): multi-regime momentum + local-vol conformal. Holdout pooled 0.2966, coverage
  0.7408. This is the champion.
- Iteration 2 (REVERTED): per-series recent-window split-conformal z. Exploded (h7 pooled -1e18).
  Lesson: bound every output; per-series calibration is dead.
- Iteration 3 (REVERTED): pooled recent-90-day split-conformal z as max with full-window z. +0.0004
  pooled, coverage 0.7408 -> 0.7380, Gate 3 fail. Lesson: recent residuals are not heavier-tailed.
- Iteration 4 (REVERTED, clean null): regime-A momentum-beta shrinkage, validation-selected. Holdout
  byte-identical to champion (s = 1.0 always). FALSIFIED: pooled OLS beta is out-of-sample optimal;
  calm-zone worst series lose on WIDTH, not q50 bias.
- Iteration 5 (REVERTED, closest near-miss, shortfall 0.0025): four bold levers measured on Discovery —
  (a) cross_series per-instance-type betas/z: wash-to-worse out-of-sample (analysis), dead;
  (b) ensemble 0.8/0.2 momentum+drift: pooled -0.0141, dead; (c) regime_change calm-vol z narrowing:
  pooled -0.0005, coverage -0.004, dead; (d) horizon_structure per-horizon drift mu_h: pooled -0.0323,
  11/11 worse, dead (in-sample mean residual is negative and does not persist). Submitted the only
  survivor: h1 tail narrowing 0.95 around q50. Discovery +0.0005 (10/11 origins better), holdout
  +0.00046 pooled, h1 +0.0047, gates 2/3/4 pass, Gate 1 shortfall 0.0025. CONFIRMED mechanism, sub-
  threshold magnitude. Lessons: h7 under-coverage is centering not width; h1 over-coverage is real and
  harvestable but worth only +0.0005; Gate 3 tolerates coverage 0.738 via the no-worse-than-champion
  clause.
