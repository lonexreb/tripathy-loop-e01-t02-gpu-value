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
- **Read `mean_q50_error`/`bias_rel` vs `mean_width_rel` before choosing a lever.** h7 under-coverage was a
  CENTERING problem (bias_rel -0.012 at h7), not a width problem; the calm-zone worst series lose on WIDTH
  (coverage 0.97), not q50 bias.

## Method
1. Diagnose from Discovery only: per-slice skill, breach asymmetry, worst series (bias vs width).
2. State ONE falsifiable hypothesis with a mechanism and a predicted sign.
3. Change one thing at a time so the ledger attributes the effect; keep everything else identical.
   (Exception: when stacking CONFIRMED levers, submit the stack and attribute via the per-lever deltas
   already measured on Holdout — iterations 6 and 8 did this cleanly.)
4. Record the result, including negative results and their magnitude. Use analyze_discovery to measure
   out-of-sample BEFORE spending a dry run or a submission (budget: ~6 analyses, 4 dry runs per iteration).
   NOTE: the analysis budget exhausts fast; plan analyses to answer several questions per call.
5. **Pseudo-origin OOS analysis (fit at origin-28d, score the next 28 days inside Discovery) is the strongest
   pre-screen available.** Iteration 8 is the decisive positive case: its 5/5-origin OOS gains transferred to
   Holdout at ~5x their Discovery magnitude. Iteration 7 is the negative case: calm-origin WIDTH-conditioning
   gains did not transfer. The difference: iteration 8's lever was an unconditional smoothing of the band
   scale (averaging over windows), iteration 7's was a state-conditioned rescale. Prefer unconditional
   smoothing over conditional rescaling.
6. **In-sample Discovery diagnostics LIE about the median when a lever widens bands**: iteration 8's
   Discovery in-sample median dropped -0.0146 while the holdout median ROSE +0.0193. The champion's
   in-sample pooled z is optimal in-sample by construction, so any widening looks bad in-sample. Trust the
   pseudo-origin OOS per-series median, not the in-sample median, when the lever changes width.

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
  recency/state signal has now failed on Holdout every time it was tried. **State/conditioned width levers
  are dead; unconditional smoothing (ensemble averaging) is the one width lever that WORKS (iteration 8).**
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
  [Iteration-8 update: the new champion sits at 0.7514 and the 2026-10-24 benchmark at 0.7526 under it;
  the binding constraint is now the Gate-3 band floor 0.74 — the "never below 0.74" clause stands.]
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
  unconditional out of sample. Only pooled/unconditional rules were ever competitive.
- Cross-collector pull (kappa > 0) and cross-AZ consensus: falsified in iteration 1. Dead levers.
- **Per-instance-type (cross_series) hierarchy is dead (iteration 5 analysis):** per-type betas/z are
  wash-to-worse out-of-sample. The pooled OLS beta is at the out-of-sample optimum.
- **h7 tail widening is dead (iterations 3, 5):** +20% h7 tails: h7 skill -0.020 despite lifting h7
  coverage 0.769 -> 0.808. h7's loss is centering, not width. [Iteration-8 note: the ensemble's
  window-averaged vol scale is NOT a fixed-factor widening and did fix h7 (+0.0133 holdout); the
  fixed-factor form remains dead.]
- **Ensemble with a drift/persistence member is dead (iteration 5):** 0.8/0.2 momentum+drift quantile
  average: pooled -0.0141, 10/11 origins worse. **Ensemble across perturbed hyperparameters of the SAME
  model is CONFIRMED-POSITIVE (iteration 8) — the distinction is that all members share the good machinery.**
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
- **Train-day ensemble members {270,540} are dead (iteration 8 analysis):** the 270-day member is
  mixed-to-negative at h7 (beta270 over-weights the 1-day lag: ~1.9 vs ~1.6 at h7). **avgbeta (averaging
  betas with a single z) is a wash — the ensemble gain lives in averaging the z/vol scale, not the point
  forecast.**
- Pool parameters across series where per-series histories are short (< 60 days).
- Point-in-time only: use rows dated on or before the origin; never hard-code dates or per-origin constants.
- Code: numpy/pandas only, vectorised, < 20 s per origin; guard empty histories and NaNs; always return
  every target series; enforce q10 <= q50 <= q90 and positivity.

## Current champion state (iteration 8, PROMOTED; builds on iterations 1, 5, 6)
Discovery pooled 0.4462, median 0.2831, coverage 0.7947, breaches 9.0%/11.5%. Holdout pooled 0.3088,
median 0.2745, coverage 0.7514, horizons 0.4030/0.3369/0.2801. Gate margins all positive
(gate1 +0.0092, gate2 +0.0243, gate3 +0.0156, gate4 +0.3001). Benchmark 2026-10-24: pooled 0.3252,
median 0.2086, coverage 0.7526.
Regimes: A AWS spot = pooled momentum OLS (train 540) + local-vol conformal with **quantiles averaged
across vol_wins {7,14,28} at h3/h7** (h1 single-window + 0.95 tail narrowing); B rate cards = empirical
change quantiles + stickiness pin + IQR width; C marketplace/non-card = trailing-60-day change quantiles
(full-history fallback), kappa = 0.

## CONFIRMED-POSITIVE levers (measured on Holdout; all four are IN the champion)
1. **h1 tail narrowing (iteration 5):** z -> z50 + 0.95*(z - z50) at h=1 only. Holdout h1 +0.0047.
   Mechanism: h1 band over-covers (0.840 vs 0.80).
2. **Regime-C recent-window change quantiles (iteration 6):** trailing 60-day h-day log-change quantiles
   (min 30 samples, full-history fallback) for non-card daily-statistics series. Holdout ~+0.0005 pooled,
   coverage +0.0028.
3. **Vol-window ensemble at h3/h7 (iteration 8, the promotion lever):** average the three quantile sets
   across vol_wins {7,14,28} (same pooled-OLS beta, train 540) at h3/h7 only. Holdout h3 +0.0124,
   h7 +0.0133, pooled +0.0122, median +0.0193, coverage +0.0106. Mechanism: the single 14-day vol window
   is an arbitrary hyperparameter; averaging over it is unconditional bagging of the band scale that
   widens h3/h7 toward the 0.80 target (0.783/0.751 -> 0.830/0.802 on Discovery) and cuts breach loss.
   Transferred at ~5x its Discovery magnitude on Holdout.
4. **Stacking confirmed levers works cleanly** (iterations 6, 8): per-lever Holdout deltas add up.

## Ranked untried directions (the lab decides stopping; these are the next hypotheses)
The champion now passes all gates with margin. Remaining headroom, ranked by expected value:
1. **ensemble: extend the vol-window averaging to h1 (currently single-window + 0.95 narrowing).**
   Iteration 8 excluded h1 because its coverage (0.803) was already at target and the full ensemble
   widened it to 0.829; but a *narrowed* h1 ensemble (average across vol_wins, then apply ~0.90-0.92
   narrowing to re-center coverage at 0.80) might beat the single-window h1. Measure OOS first; the risk
   is repeating the h1-widening median damage that made the full ensemble fail Gate-2 screening.
2. **ensemble: richer member set at h3/h7** (e.g., vol_wins {7,14,21,28} or weighted blend 0.25/0.5/0.25
   centred on 14). The pseudo-origin analysis showed the half-weighted blend ~equal to the plain average;
   a 4-member set is the natural next perturbation. Cheap dry run; small expected gain.
3. **horizon_structure: iterated 1-day forecasts (h3 = 3 chained h1 steps with variance accumulation).**
   Genuinely different q50/width generator; the h1 machinery is the best-calibrated. Bold; measure OOS
   first — chaining compounds beta error.
4. **exogenous_signals: hardware_fundamentals / capability_ladder level anchoring for the marketplace and
   non-AWS hyperscaler q50 (bias_rel -0.05/-0.06 at h3/h7).** Small slice (~4% of loss) but the bias is
   real and persistent; check whether the ladder maps to the benchmark gpu_models before spending anything.
5. **robustness: log-band clip (last log value +/- 1.0) bundled into the next candidate.** Pure safety,
   expected NO_OP on scored cells; bundle, never submit alone.

## Notebook (append-only ledger of what was tried)
- Iteration 0 (SEED): naive baseline, holdout pooled 0.2306.
- Iteration 1 (PROMOTED): multi-regime momentum + local-vol conformal. Holdout pooled 0.2966, coverage
  0.7408. The champion for iterations 2-7.
- Iteration 2 (REVERTED): per-series recent-window split-conformal z. Exploded (h7 pooled -1e18).
  Lesson: bound every output; per-series calibration is dead.
- Iteration 3 (REVERTED): pooled recent-90-day split-conformal z as max with full-window z. +0.0004
  pooled, coverage 0.7408 -> 0.7380, Gate 3 fail. Lesson: recent residuals are not heavier-tailed.
- Iteration 4 (REVERTED, clean null): regime-A momentum-beta shrinkage, validation-selected. Holdout
  byte-identical to champion (s = 1.0 always). FALSIFIED: pooled OLS beta is out-of-sample optimal.
- Iteration 5 (REVERTED, Gate 1 shortfall 0.0025): four bold levers measured on Discovery — cross_series
  per-type hierarchy dead; ensemble momentum+drift dead (-0.0141); calm-vol narrowing dead (-0.0005);
  per-horizon drift mu_h dead (-0.0323). Survivor: h1 tail narrowing 0.95. Holdout +0.00046 pooled,
  h1 +0.0047. CONFIRMED mechanism, sub-threshold alone.
- Iteration 6 (REVERTED, Gate 1 shortfall 0.002, builds on 5): measured exogenous market-momentum OOS
  (negative, dead), long-lag momentum (dry run -0.0050, dead), regime-B spread floor (dry run -0.0118,
  dead). Submitted the stack of the two confirmed levers: regime-C 60-day change quantiles + h1
  narrowing. Holdout pooled 0.2976 (+0.000953), median 0.2586, coverage 0.7436. Gates 2/3/4 pass,
  Gate 1 shortfall 0.002. Both levers confirmed positive on Holdout.
- Iteration 7 (REVERTED, Gate 1 shortfall 0.0027, builds on 6, regime_change bold): pseudo-origin OOS
  analysis falsified z-centering (no-op), h7 biasfix (origin-dependent), h7 beta damping
  (wash-to-worse), regime-C q50 centering (wash), aggregator_medians/index_prints (degenerate), and ALL
  per-series conditional width rules. Survivor: pooled vol-state tail scaling. Discovery +0.003766
  (9/11 origins) but Holdout +0.000259 only (the state lever cost ~-0.0007 vs the stack). FALSIFIED:
  calm-origin width-conditioning gains do not transfer. Width levers via state conditioning exhausted.
- Iteration 8 (PROMOTED, ensemble bold, builds on 6): vol-window parameter-perturbation ensemble.
  Pseudo-origin OOS analysis (fit at origin-28d, score next 28 days, 5 Discovery origins): vol-window
  ensemble {7,14,28} beats the single 14-day member at h3 (5/5 origins, avg +0.0066) and h7 (5/5, avg
  +0.0064); train-day member {270,540} negative at h7 (excluded); avgbeta a wash (excluded — the gain
  is in averaging the z/vol scale, not the beta); per-series median skill OOS improves 4/5 origins.
  First dry run (full ensemble incl. h1): +0.0038 pooled but median -0.0131 and h1 coverage 0.803 ->
  0.829 (away from target) — Gate-2 risk. Second dry run (h3/h7 only, h1 byte-identical to the stack):
  pooled +0.003335, h3 +0.0022, h7 +0.0036, coverage +0.0152 toward 0.80, q10 breach 10.1% -> 9.0%.
  Submitted. HOLDOUT: pooled 0.3088 (+0.012226), median 0.2745 (+0.0193), coverage 0.7514 (+0.0106),
  h1 +0.0047 / h3 +0.0124 / h7 +0.0133. ALL 4 GATES PASS (margins +0.0092/+0.0243/+0.0156/+0.3001).
  The h3/h7 holdout deltas transferred at ~5x their Discovery magnitude. LESSONS: (a) unconditional
  smoothing of the band scale transfers; state-conditioned rescaling does not; (b) in-sample Discovery
  median drops when widening (leakage: in-sample z is optimal in-sample) while the OOS/holdout median
  RISES — trust the pseudo-origin OOS median, not the in-sample one; (c) restrict ensemble members to
  horizons where the champion under-covers (h3 0.783, h7 0.751), keep calibrated horizons (h1 0.803)
  on their confirmed single-window machinery.
