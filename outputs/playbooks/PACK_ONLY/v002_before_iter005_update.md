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

## HARD guardrails (all measured; cite the falsifying iteration to remove any of them)
- **Bound every output.** Clip per-series z to a small multiple of pooled z AND clip final log-quantiles to
  a sane band around the last value (log q within +/- 1 of log last). Iteration 2's explosion (h7 pooled
  -1e18) came from unbounded per-series z on a sticky series.
- **Never touch the LOWER quantile with a "max".** Iteration 3 falsified "elementwise max of pooled z is
  widening-only": max widens the upper side but RAISES q10, narrowing the lower band. Result: q10 breach
  10.1% -> 10.9%, holdout coverage 0.7408 -> 0.7380, Gate 3 fail. Any upper-tail widening experiment must
  leave z10 byte-identical to the champion.
- **Recent-window residuals do NOT estimate forward tails (falsified twice: iterations 2 and 3).**
  Iteration 2: per-series recent-window z narrowed bands and exploded. Iteration 3: pooled split-conformal
  recent-90-day z, even as max with the full-window z, moved q90 breach only 11.9% -> 11.6% and gained
  +0.0004 pooled while costing -0.0028 coverage. The champion's full-window in-sample pooled z is already
  honest on the upper side.
- **Validation-selected parameter changes that can also select "no change" tend to degenerate to the
  champion on holdout** (iteration 4: shrinkage s=1.0 chosen at every holdout origin -> holdout aggregates
  byte-identical -> Gate 1 fails by construction). If a lever needs a *gain*, it must change something
  unconditionally or its selection rule must be shown (on Discovery) to actually fire.
- **Holdout coverage 0.7408 sits at the Gate-3 edge.** Iteration 5 showed Gate 3 still passes at 0.738 via
  the "no worse than champion's distance" clause, but treat any coverage drop as near-fatal:
  coverage-neutral-or-widening changes only; never narrow aggregate coverage.
- **In-sample mean residuals do NOT persist forward (iteration 5).** Adding the pooled per-horizon mean
  residual mu_h to all quantiles: pooled -0.0323, 11/11 origins worse. The forward bias_rel
  (-0.001/-0.006/-0.013 by horizon) is NOT an intercept problem; do not "fix" bias with in-sample drift.
- **Per-series local calibration is catastrophic (iteration 2). Dead lever.**
- **Per-instance-type (cross_series) hierarchy is dead (iteration 5 analysis):** per-type betas/z are
  wash-to-worse out-of-sample (q50 err 0.01136 -> 0.01149 at h1; pinball worse on 2 of 3 origins at h7).
  The pooled OLS beta is at the out-of-sample optimum.
- **Momentum-beta shrinkage toward zero, validation-selected (iteration 4): DEAD.** Holdout aggregates
  byte-identical to champion; on Discovery it fired only at h1 and slightly hurt. The pooled OLS beta is
  already out-of-sample optimal.
- **Cross-collector pull (kappa > 0) and cross-AZ consensus: falsified in iteration 1. Dead levers**
  (for q50; cross-collector WIDTH conditioning is untested — see ideas).
- **Ensemble with a drift/persistence member is dead (iteration 5):** 0.8/0.2 momentum+drift quantile
  average: pooled -0.0141, 10/11 origins worse. The drift member degrades q50 and its wider tails cost
  more than the breach saving.
- **Calm-regime (bottom-quartile local vol) z narrowing is dead (iteration 5):** pooled -0.0005,
  coverage -0.004. The calm-zone over-coverage cannot be harvested by conditional narrowing.
- **h7 tail widening is dead (iterations 3 and 5):** +20% h7 tails: h7 skill -0.020 despite lifting h7
  coverage 0.769 -> 0.808. h7's loss is centering, not width.
- Point-in-time only: use rows dated on or before the origin; never hard-code dates or per-origin
  constants tuned to a single origin.
- Pool parameters across series where per-series histories are short (< 60 days).
- Code: numpy/pandas only, vectorised, < 20 s per origin; guard empty histories and NaNs; always return
  every target series; enforce q10 <= q50 <= q90 and positivity.

## Current champion state (iteration 1, promoted; iterations 2, 3, 4, 5 reverted against it)
Discovery pooled 0.4429, median 0.2977, coverage 0.7795, breaches 10.1%/11.9% (upper tail slightly
under-covered). Holdout pooled 0.2966, median 0.2552, coverage 0.7408, horizons 0.398/0.324/0.267.
Regimes: A AWS spot = pooled momentum OLS on 1/3/7-day log changes + local-vol conformal (full-window
in-sample pooled z); B rate cards = empirical change quantiles + stickiness pin + IQR width; C
marketplace = drift, cross-collector pull disabled (kappa = 0, falsified in iteration 1).

## Confirmed-positive but sub-threshold lever (iteration 5, the closest near-miss)
**h1 tail narrowing (z -> z50 + 0.95*(z - z50) at h=1 only, q50/h3/h7 byte-identical): Discovery pooled
+0.0005 (10/11 origins better), Holdout pooled +0.00046, h1 skill +0.0047, gates 2/3/4 passed, Gate 1
shortfall 0.0025.** The mechanism (h1 band over-covers: 0.840 vs 0.80 target; the 0.1-weighted width cost
on inside cells exceeds the breach saving) is CONFIRMED on both Discovery and Holdout. It cannot pass
Gate 1 alone (+0.003 needed). Stack it with a second independent positive lever; do not re-submit standalone.

## Ranked untried directions (the lab decides stopping; these are the next hypotheses)
1. **exogenous_signals (UNTRIED, top priority): the pack tables the champion ignores.** Regime A uses only
   series_history. daily_statistics (p25/median/p75/min per collector), aggregator_medians (weekly),
   index_prints, hardware_fundamentals, capability_ladder/capability_units are all unused. Hypotheses:
   (a) cross-collector dispersion (p75-p25)/median of the SAME series conditions forward tail width better
   than the 14-day |1-day log change| vol (contemporaneous demand-supply spread vs smoothed price history);
   (b) index_prints / hardware fundamentals give a level anchor for the h7 centering bias (bias_rel -0.012).
   Measure out-of-sample with analyze_discovery FIRST (iteration-5 protocol: train on d <= od-90, test on
   the recent 90 days, pooled pinball vs champion machinery) before any dry run.
2. **Stack the confirmed h1 narrowing (+0.0005) with the best exogenous lever** if (1) measures >= +0.0025
   out-of-sample; the sum must clear +0.003 on holdout. Both levers must fire unconditionally.
3. **robustness (untried):** add the log-band clip (last log value +/- 1.0) as a pure safety change.
   Expected NO_OP on holdout (never binds on sane series) -> Gate 1 fail by construction; only worth
   bundling with (1)/(2).
4. **regime_change (partially burned):** calm-regime narrowing falsified (iteration 5); jump/step detection
   on sticky administered AWS spot prices is the remaining untested variant, but the prior after five
   falsified calibration levers is strongly negative.
5. **horizon_structure (partially burned):** h1 narrowing confirmed sub-threshold; h7 widening and drift
   both falsified. Direct-vs-iterated multi-step (e.g. iterate the h1 model 3/7 times with accumulated z)
   is untested but the h7 centering bias suggests iteration would amplify, not fix, the bias.

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
- Iteration 5 (REVERTED, closest near-miss, Gate 1 shortfall 0.0025): four bold levers measured on
  Discovery — (a) cross_series per-instance-type betas/z: wash-to-worse out-of-sample (analysis), dead;
  (b) ensemble 0.8/0.2 momentum+drift: pooled -0.0141, dead; (c) regime_change calm-vol z narrowing:
  pooled -0.0005, coverage -0.004, dead; (d) horizon_structure per-horizon drift mu_h: pooled -0.0323,
  11/11 worse, dead (in-sample mean residual is negative and does not persist). Submitted the only
  survivor: h1 tail narrowing 0.95 around q50. Discovery +0.0005 (10/11 origins better), holdout
  +0.00046 pooled, h1 +0.0047, gates 2/3/4 pass, Gate 1 shortfall 0.0025. CONFIRMED mechanism, sub-
  threshold magnitude. Lessons: h7 under-coverage is centering not width; h1 over-coverage is real and
  harvestable but worth only +0.0005; Gate 3 tolerates coverage 0.738 via the no-worse-than-champion clause.
