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
   (Exception: when stacking CONFIRMED levers, submit the stack and attribute via the per-lever deltas
   already measured on Holdout — iteration 6 did this cleanly.)
4. Record the result, including negative results and their magnitude. Use analyze_discovery to measure
   out-of-sample BEFORE spending a dry run or a submission (budget: ~6 analyses, 4 dry runs per iteration).
   NOTE: the analysis budget exhausts fast (6 calls); plan analyses to answer several questions per call.

## HARD guardrails (all measured; cite the falsifying iteration to remove any of them)
- **Bound every output.** Clip per-series z to a small multiple of pooled z AND clip final log-quantiles to
  a sane band around the last value (log q within +/- 1 of log last). Iteration 2's explosion (h7 pooled
  -1e18) came from unbounded per-series z on a sticky series.
- **Never touch the LOWER quantile with a "max".** Iteration 3 falsified "elementwise max of pooled z is
  widening-only": max widens the upper side but RAISES q10, narrowing the lower band. Result: q10 breach
  10.1% -> 10.9%, holdout coverage 0.7408 -> 0.7380, Gate 3 fail. Any upper-tail widening experiment must
  leave z10 byte-identical to the champion.
- **Recent-window residuals do NOT estimate forward tails (falsified twice).** Iteration 2: per-series
  recent-window split-conformal z narrowed bands and exploded. Iteration 3: pooled split-conformal
  recent-90-day z, even taken as max with the full-window z, moved q90 breach only 11.9% -> 11.6% and
  gained just +0.0004 pooled. The champion's full-window in-sample pooled z is already honest on the
  upper side. (Regime C is the exception: iteration 6's 60-day change-quantile window HELPED on holdout.)
- **Validation-selected parameter changes that can also select "no change" tend to degenerate to the
  champion on holdout** (iteration 4: shrinkage s=1.0 chosen at every holdout origin -> holdout aggregates
  byte-identical -> Gate 1 fails by construction). If a lever needs a *gain*, it must change something
  unconditionally or its selection rule must be shown (on Discovery) to actually fire.
- **Holdout coverage 0.7408 was 0.0008 above the Gate-3 floor; iteration 6's stack moved it to 0.7436
  (in-band).** Coverage-neutral-or-widening changes only; never narrow aggregate coverage below ~0.74.
- Point-in-time only: use rows dated on or before the origin; never hard-code dates or per-origin constants.
- Upper-tail widening via max(full-window, recent split-conformal) z: +0.0004 pooled, -0.0028 coverage.
  Dead lever — do not retry recent-window conformal in any form.
- Per-series local calibration: catastrophic (iteration 2). Dead lever.
- **Momentum-beta shrinkage toward zero, validation-selected (iteration 4): DEAD.** Holdout aggregates
  byte-identical to champion (validation always picked s = 1.0); where it fired on Discovery h1 it
  slightly hurt. The pooled OLS beta is already at the out-of-sample optimum.
- Cross-collector pull (kappa > 0) and cross-AZ consensus: falsified in iteration 1. Dead levers.
- **Per-instance-type (cross_series) hierarchy is dead (iteration 5 analysis):** per-type betas/z are
  wash-to-worse out-of-sample. The pooled OLS beta is at the out-of-sample optimum.
- **h7 tail widening is dead (iterations 3, 5):** +20% h7 tails: h7 skill -0.020 despite lifting h7
  coverage 0.769 -> 0.808. h7's loss is centering, not width.
- **In-sample mean residuals do NOT persist forward (iteration 5).** Adding the pooled per-horizon mean
  residual mu_h to all quantiles: pooled -0.0323, 11/11 origins worse. Do not "fix" bias with in-sample drift.
- **Ensemble with a drift/persistence member is dead (iteration 5):** 0.8/0.2 momentum+drift quantile
  average: pooled -0.0141, 10/11 origins worse.
- **Calm-regime (bottom-quartile local vol) z narrowing is dead (iteration 5):** pooled -0.0005,
  coverage -0.004.
- **Exogenous market-median momentum is dead (iteration 6 analysis):** augmenting the pooled momentum OLS
  with daily_statistics.hyperscaler_spot market-wide per-gpu_model 1/3/7-day log changes (afhubbard
  collector, maps to A100 40GB/80GB, H100, H200) is NEGATIVE out-of-sample at both tested origins:
  MSE +3-10% worse at every horizon. The market signal is lagged and noisier than the per-AZ price.
- **Long-lag momentum (14/30-day log changes) is dead (iteration 6 dry run):** pooled -0.0050, 7/11
  origins worse, all horizons worse, despite a mixed h7 MSE signal in analysis. In-sample MSE gains do
  not transfer.
- **Regime-B rate-card spread floor is dead (iteration 6 dry run):** a widening-only log-spread floor of
  0.05*sqrt(h) on rate cards: pooled -0.0118, 8/11 origins worse, rate-card slice skill -0.79 to -1.03.
  The EV-per-unit-width argument (breach 0.29 -> +0.21/unit) was WRONG: rate-card breaches are
  cent-level absolute moves, so a 5% relative band dwarfs them. Compute EV against ABSOLUTE breach loss,
  not breach rate, when the series' own vol is orders of magnitude below the floor.
- Pool parameters across series where per-series histories are short (< 60 days).
- Code: numpy/pandas only, vectorised, < 20 s per origin; guard empty histories and NaNs; always return
  every target series; enforce q10 <= q50 <= q90 and positivity.

## Current champion state (iteration 1, promoted; iterations 2-6 reverted against it)
Discovery pooled 0.4429, median 0.2977, coverage 0.7795, breaches 10.1%/11.9%. Holdout pooled 0.2966,
median 0.2552, coverage 0.7408, horizons 0.398/0.324/0.267. Regimes: A AWS spot = pooled momentum OLS on
1/3/7-day log changes + local-vol conformal (full-window in-sample pooled z); B rate cards = empirical
change quantiles + stickiness pin + IQR width; C marketplace/non-card = drift, kappa = 0.

## CONFIRMED-POSITIVE levers (measured on BOTH Discovery and Holdout; stack, never submit alone)
1. **h1 tail narrowing (iteration 5):** z -> z50 + 0.95*(z - z50) at h=1 only. Discovery +0.0005 pooled,
   Holdout +0.00046 pooled, h1 +0.0047. Mechanism: h1 band over-covers (0.840 vs 0.80).
2. **Regime-C recent-window change quantiles (iteration 6):** trailing 60-day h-day log-change quantiles
   (min 30 samples, full-history fallback) for non-card daily-statistics series. Holdout contribution
   ~+0.0005 pooled, coverage +0.0028 (0.7408 -> 0.7436, in-band), h7 +0.0009. ~Neutral on Discovery
   (recent-60 quantiles close to full-history there) but positive out of sample.
   Stack of 1+2 (iteration 6): Holdout pooled 0.2976 (+0.000953), median 0.2586 (+0.0033), coverage
   0.7436, h1 +0.0047 / h3 -0.0002 / h7 +0.0009. Gates 2/3/4 pass; **Gate 1 shortfall 0.002**
   (need +0.003 over champion, got +0.000953). This is the closest near-miss; build on iteration 6.

## Ranked untried directions (the lab decides stopping; these are the next hypotheses)
The remaining Gate-1 gap after stacking both confirmed levers is ~0.002 pooled on Holdout. Ideas ranked
by expected value; measure out-of-sample with analyze_discovery FIRST:
1. **horizon_structure: h3 tail narrowing (untested, same mechanism as the confirmed h1 lever).**
   Discovery h3 coverage 0.783 (vs 0.80 target) is much closer to calibrated than h1's 0.840, so the
   expected gain is smaller than h1's +0.005 h1-skill, but h3 carries 25% of pooled loss. Try z3 ->
   z50 + 0.98*(z3 - z50) (or 0.95) at h=3 only, stacked on the iteration-6 candidate. Dry-run to pick
   the scale; the h1 precedent says the width cost on inside cells is the risk.
2. **regime_change: extend regime-C recency to the sticky pin / q50.** The 60-day window helped the
   band; the q50 pin (zero_share >= 0.5 -> q50 = last value) and the marketplace q50 (bias_rel -0.05 at
   h3, -0.06 at h7 on Discovery) are untouched. A recent-window median change (drift of the last 60
   days) added to q50 for non-card series is the natural next lever; measure OOS first.
3. **exogenous_signals: market DISPERSION conditioning (untested; momentum form is dead).**
   daily_statistics.hyperscaler_spot gives (p75-p25)/median per gpu_model per day; test whether it
   conditions the AWS forward tail width beyond the 14-day own-vol. The momentum form was negative, so
   prior is weak; only pursue if the analysis shows corr(disp, |z|) clearly > 0.1 at both origins.
4. **exogenous_signals: aggregator_medians (weekly, 13,781 rows) for regime C/B.** Untouched table;
   weekly provider medians could anchor the marketplace q50 level. Small slice (~4% of loss).
5. **robustness: log-band clip (last log value +/- 1.0).** Pure safety, expected NO_OP; bundle only.
6. **regime_change on regime A (jump/step detection on sticky administered prices):** prior strongly
   negative after six falsified calibration levers; last resort.

## Notebook (append-only ledger of what was tried)
- Iteration 0 (SEED): naive baseline, holdout pooled 0.2306.
- Iteration 1 (PROMOTED): multi-regime momentum + local-vol conformal. Holdout pooled 0.2966, coverage
  0.7408. This is the champion.
- Iteration 2 (REVERTED): per-series recent-window split-conformal z. Exploded (h7 pooled -1e18).
  Lesson: bound every output; per-series calibration is dead.
- Iteration 3 (REVERTED): pooled recent-90-day split-conformal z as max with full-window z. +0.0004
  pooled, coverage 0.7408 -> 0.7380, Gate 3 fail. Lesson: recent residuals are not heavier-tailed.
- Iteration 4 (REVERTED, clean null): regime-A momentum-beta shrinkage, validation-selected. Holdout
  byte-identical to champion (s = 1.0 always). FALSIFIED: pooled OLS beta is out-of-sample optimal.
- Iteration 5 (REVERTED, Gate 1 shortfall 0.0025): four bold levers measured on Discovery — cross_series
  per-type hierarchy dead; ensemble momentum+drift dead (-0.0141); calm-vol narrowing dead (-0.0005);
  per-horizon drift mu_h dead (-0.0323). Survivor: h1 tail narrowing 0.95. Holdout +0.00046 pooled,
  h1 +0.0047, gates 2/3/4 pass, Gate 1 shortfall 0.0025. CONFIRMED mechanism, sub-threshold alone.
- Iteration 6 (REVERTED, closest near-miss, Gate 1 shortfall 0.002, builds on 5): measured exogenous
  market-momentum OOS (negative, dead), long-lag momentum (dry run -0.0050, dead), regime-B spread floor
  (dry run -0.0118, dead; EV argument wrong for cent-level series). Submitted the stack of the two
  confirmed levers: regime-C 60-day change quantiles + h1 narrowing. Discovery +0.000369 (8/11 origins
  better), Holdout pooled 0.2976 (+0.000953), median 0.2586 (+0.0033), coverage 0.7436 (+0.0028,
  in-band), h1 +0.0047 / h7 +0.0009. Gates 2/3/4 pass, Gate 1 shortfall 0.002. Both levers confirmed
  positive on Holdout; the stack is the new base to build on. Lessons: (a) regime C responds to
  recency out of sample even when Discovery shows ~nothing; (b) the analysis budget (6) exhausts
  before the dry-run budget (4) — batch questions per analysis call; (c) in-sample MSE gains (long
  lags) do not transfer to the pinball screen.
