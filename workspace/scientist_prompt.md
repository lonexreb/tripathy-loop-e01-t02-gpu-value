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
- **Read `mean_q50_error`/`bias_rel` vs `mean_width_rel` before choosing a lever.** h7 under-coverage is a
  CENTERING problem (bias_rel -0.012 at h7, growing with horizon), not a width problem; the calm-zone worst
  series lose on WIDTH (coverage 0.97), not q50 bias.

## Method
1. Diagnose from Discovery only: per-slice skill, breach asymmetry, worst series (bias vs width).
2. State ONE falsifiable hypothesis with a mechanism and a predicted sign.
3. Change one thing at a time so the ledger attributes the effect; keep everything else identical.
   (Exception: when stacking CONFIRMED levers, submit the stack and attribute via the per-lever deltas
   already measured on Holdout — iterations 6 and 7 did this cleanly.)
4. Record the result, including negative results and their magnitude. Use analyze_discovery to measure
   out-of-sample BEFORE spending a dry run or a submission (budget: ~6 analyses, 4 dry runs per iteration).
   NOTE: the analysis budget exhausts fast; plan analyses to answer several questions per call.
5. **Pseudo-origin OOS analysis (fit at origin-28d, score the next 28 days inside Discovery) is the strongest
   pre-screen available** — but iteration 7 showed its calm-origin width gains do NOT transfer to Holdout.
   Treat pseudo-origin gains as necessary, not sufficient; weight Holdout evidence from prior iterations higher.

## HARD guardrails (all measured; cite the falsifying iteration to remove any of them)
- **Bound every output.** Clip per-series z to a small multiple of pooled z AND clip final log-quantiles to
  a sane band around the last value (log q within +/- 1 of log last). Iteration 2's explosion (h7 pooled
  -1e18) came from unbounded per-series z on a sticky series.
- **Never touch the LOWER quantile with a "max".** Iteration 3 falsified "elementwise max of pooled z is
  widening-only": max widens the upper side but RAISES q10, narrowing the lower band. Result: q10 breach
  10.1% -> 10.9%, holdout coverage 0.7408 -> 0.7380, Gate 3 fail. Any upper-tail widening experiment must
  leave z10 byte-identical to the champion.
- **Recent-window residuals do NOT estimate forward tails (falsified three times).** Iterations 2, 3 (recent
  split-conformal z) and 7 (pooled vol-state tail scaling: Discovery +0.0034, Holdout ~-0.0007 marginal).
  The champion's full-window in-sample pooled z is honest in expectation; conditioning its WIDTH on any
  recency/state signal has now failed on Holdout every time it was tried. **Width levers are exhausted.**
- **Upper-tail widening via max(full-window, recent split-conformal) z: +0.0004 pooled, -0.0028 coverage.
  Dead lever — do not retry recent-window conformal in any form** (iterations 3, 5).
- **Per-series local calibration: catastrophic (iteration 2). Dead lever.**
- **Validation-selected parameter changes that can also select "no change" tend to degenerate to the
  champion on holdout** (iteration 4: shrinkage s=1.0 chosen at every holdout origin -> holdout aggregates
  byte-identical -> Gate 1 fails by construction). If a lever needs a *gain*, it must change something
  unconditionally or its selection rule must be shown (on Discovery) to actually fire.
- **Holdout coverage must stay >= 0.7358** (champion 0.7408, Gate-3 distance clause floor; the 2026-10-24
  benchmark sits at 0.7320). The iteration-6 stack sits at 0.7436 (in-band); iteration 7's variant at
  0.7429. Coverage-neutral-or-widening changes only; never narrow aggregate coverage below 0.74.
- **Momentum-beta shrinkage toward zero, validation-selected (iteration 4): DEAD.** Holdout aggregates
  byte-identical to champion (validation always picked s = 1.0); where it fired on Discovery h1 it
  slightly hurt. The pooled OLS beta is already at the out-of-sample optimum.
- **Momentum-beta damping toward zero, unconditional (iteration 7): DEAD.** 0.85/0.70 damping of the h7
  momentum term is wash-to-worse out-of-sample in the pseudo-origin analysis. The pooled OLS beta is at
  the out-of-sample optimum in both selection and unconditional forms.
- **In-sample mean residuals do NOT persist forward (iterations 5, 7).** Adding the pooled per-horizon mean
  residual mu_h to all quantiles: pooled -0.0323, 11/11 origins worse. The h7 standardised mean error is
  -0.9..-1.2 std but subtracting it is origin-dependent (+0.27 at one pseudo-origin, -0.25 at another).
  Do not "fix" bias with in-sample drift.
- **Per-series conditional width rules are dead (iteration 7 analysis):** calm-only 0.95, continuous
  clip(vol-ratio, 0.93, 1.07), 0.95*vol-ratio — every cross-sectional conditional is worse than
  unconditional out of sample. Only pooled/unconditional rules were ever competitive, and those fail on
  Holdout (see the vol-state guardrail above).
- Cross-collector pull (kappa > 0) and cross-AZ consensus: falsified in iteration 1. Dead levers.
- **Per-instance-type (cross_series) hierarchy is dead (iteration 5 analysis):** per-type betas/z are
  wash-to-worse out-of-sample. The pooled OLS beta is at the out-of-sample optimum.
- **h7 tail widening is dead (iterations 3, 5):** +20% h7 tails: h7 skill -0.020 despite lifting h7
  coverage 0.769 -> 0.808. h7's loss is centering, not width.
- **Ensemble with a drift/persistence member is dead (iteration 5):** 0.8/0.2 momentum+drift quantile
  average: pooled -0.0141, 10/11 origins worse.
- **Calm-regime (bottom-quartile local vol) z narrowing is dead (iteration 5):** pooled -0.0005,
  coverage -0.004.
- **Exogenous market-median momentum is dead (iteration 6 analysis):** daily_statistics.hyperscaler_spot
  market-wide per-gpu_model 1/3/7-day log changes are NEGATIVE out-of-sample (MSE +3-10% worse at every
  horizon). The market signal is lagged and noisier than the per-AZ price.
- **Long-lag momentum (14/30-day log changes) is dead (iteration 6 dry run):** pooled -0.0050, 7/11
  origins worse. In-sample MSE gains do not transfer.
- **Regime-B rate-card spread floor is dead (iteration 6 dry run):** pooled -0.0118, 8/11 origins worse.
  Compute EV against ABSOLUTE breach loss, not breach rate, when the series' own vol is orders of
  magnitude below the floor.
- **aggregator_medians and index_prints are degenerate for PACK_ONLY (iteration 7 analysis):**
  aggregator_medians is a single source (getdeploying_weekly, reservation-style weekly medians) with no
  overlap to the benchmark series' collectors; index_prints has 0-12 rows. Dead as exogenous signals here.
- **Regime-C q50 centering variants are dead (iteration 7 analysis):** recent-60 median drift and
  zero-pin are a wash on the 4-5 evaluable series.
- Pool parameters across series where per-series histories are short (< 60 days).
- Point-in-time only: use rows dated on or before the origin; never hard-code dates or per-origin constants.
- Code: numpy/pandas only, vectorised, < 20 s per origin; guard empty histories and NaNs; always return
  every target series; enforce q10 <= q50 <= q90 and positivity.

## Current champion state (iteration 1, promoted; iterations 2-7 reverted against it)
Discovery pooled 0.4429, median 0.2977, coverage 0.7795, breaches 10.1%/11.9%. Holdout pooled 0.2966,
median 0.2552, coverage 0.7408, horizons 0.398/0.324/0.267. Regimes: A AWS spot = pooled momentum OLS on
1/3/7-day log changes + local-vol conformal (full-window in-sample pooled z); B rate cards = empirical
change quantiles + stickiness pin + IQR width; C marketplace/non-card = drift, kappa = 0.

## CONFIRMED-POSITIVE levers (measured on BOTH Discovery and Holdout; stack, never submit alone)
1. **h1 tail narrowing (iteration 5):** z -> z50 + 0.95*(z - z50) at h=1 only. Discovery +0.0005 pooled,
   Holdout +0.00046 pooled, h1 +0.0047. Mechanism: h1 band over-covers (0.840 vs 0.80).
2. **Regime-C recent-window change quantiles (iteration 6):** trailing 60-day h-day log-change quantiles
   (min 30 samples, full-history fallback) for non-card daily-statistics series. Holdout contribution
   ~+0.0005 pooled, coverage +0.0028 (0.7408 -> 0.7436, in-band), h7 +0.0009. ~Neutral on Discovery but
   positive out of sample.
   Stack of 1+2 (iteration 6): Holdout pooled 0.2976 (+0.000953), median 0.2586 (+0.0033), coverage
   0.7436, h1 +0.0047 / h3 -0.0002 / h7 +0.0009. Gates 2/3/4 pass; Gate 1 shortfall 0.002.
   **This stack is the best measured base; every future candidate should build on iteration 6.**

## FALSIFIED on Holdout (iteration 7, do not retry)
- **Pooled vol-state tail scaling at h3/h7** (market-wide median of current-14d-vol / own-90d-median-vol
  < 1 -> narrow tails by clip(ratio, {3:0.95, 7:0.93}, 1.0)): Discovery +0.0034 marginal (9/11 origins
  better, h7 slice coverage cost 0.0), but Holdout marginal ~-0.0007 pooled (total +0.000259 vs the
  stack's +0.000953; h3 -0.0006, h7 -0.0001). The Holdout origins (2026-09-30..10-24) did not reward
  calm-market narrowing even though 9/11 Discovery origins did. Gates 2/3/4 passed (coverage 0.7429
  in-band); Gate 1 shortfall 0.0027. Lesson: pseudo-origin OOS gains in WIDTH do not transfer; the width
  surface is exhausted.

## Ranked untried directions (the lab decides stopping; these are the next hypotheses)
The Gate-1 gap is +0.002 over the champion (need pooled > 0.2996; the iteration-6 stack delivers 0.2976).
Width levers are exhausted (see guardrails); the remaining headroom must come from q50/centering or from
smoothing, not from band shape. Ideas ranked by expected value:
1. **ensemble: parameter-perturbation quantile averaging (untried family variant).** Average the three
   quantiles across perturbed hyperparameters of the CONFIRMED stack (e.g., vol_win 7/14/28, or
   train_days 270/540) — a poor-man's bagging that fires unconditionally, changes q50 AND width slightly,
   and cannot explode (all members are bounded by the same machinery). The momentum+drift ensemble was
   negative because the drift member was bad; a same-model-different-window ensemble has no such member.
   Measure OOS with pseudo-origin analysis first; the risk is that averaging widens q50 error on trending series.
2. **exogenous_signals: hardware_fundamentals / capability_ladder / capability_units level anchoring
   (untouched tables).** The marketplace q50 (bias_rel -0.05 at h3, -0.06 at h7) and the non-AWS
   hyperscaler q50 could be anchored to a capability-adjusted cross-GPU level (price per capability unit).
   Small slice (~4% of loss) but the bias is real and persistent; check whether the ladder maps to the
   benchmark gpu_models at all before spending anything.
3. **horizon_structure: iterated 1-day forecasts (h3 = 3 chained h1 steps).** The h1 machinery is the
   best-calibrated (coverage 0.803 after narrowing); iterating it to h3/h7 with variance accumulation is
   a genuinely different q50/width generator, not a width rescale. Bold; measure OOS first — the momentum
   OLS at h1 chained 3/7 times may compound beta error.
4. **robustness: log-band clip (last log value +/- 1.0) bundled into the next candidate.** Pure safety,
   expected NO_OP on scored cells; bundle, never submit alone.
5. **regime_change on regime A (jump/step detection on sticky administered prices):** prior strongly
   negative after seven falsified calibration levers; last resort.

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
  (dry run -0.0118, dead). Submitted the stack of the two confirmed levers: regime-C 60-day change
  quantiles + h1 narrowing. Holdout pooled 0.2976 (+0.000953), median 0.2586 (+0.0033), coverage 0.7436
  (in-band), h1 +0.0047 / h7 +0.0009. Gates 2/3/4 pass, Gate 1 shortfall 0.002. Both levers confirmed
  positive on Holdout; the stack is the base to build on.
- Iteration 7 (REVERTED, Gate 1 shortfall 0.0027, builds on 6, regime_change bold): pseudo-origin OOS
  analysis (3 analyses) falsified z-centering (no-op), h7 biasfix (origin-dependent), h7 beta damping
  (wash-to-worse), regime-C q50 centering (wash), aggregator_medians/index_prints (degenerate tables),
  and ALL per-series conditional width rules. Survivor: pooled vol-state tail scaling (market-wide
  median vol-ratio < 1 -> narrow h3 by 0.95, h7 by 0.93; h1 excluded after per-horizon decomposition).
  Discovery +0.003766 (9/11 origins better, h7 slice coverage cost 0.0). Holdout: pooled 0.2969
  (+0.000259 vs champion = the state lever cost ~-0.0007 vs the stack), coverage 0.7429 (in-band),
  h1 +0.0047 / h3 -0.0006 / h7 -0.0001. Gates 2/3/4 pass, Gate 1 shortfall 0.0027. FALSIFIED: the
  calm-origin width gains do not transfer to Holdout. Width levers are now exhausted with Holdout
  evidence; future gains must come from q50/centering or smoothing (see ranked directions).
