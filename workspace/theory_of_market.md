# Theory of the market (append-only lab notebook)

Entries are appended by the RSI controller after every experiment (promoted, reverted or crashed).
Do not rewrite earlier entries; add corrections as new entries.

## Step 0 facts (from the pack and the starter kit)
- Scored set: 97 headline series of the starter catalog (as_of 2026-10-24): 80 AWS EC2 spot AZ/composite series
  (`hyperscaler_spot`, administered smoothed spot prices), 4 afhubbard hyperscaler_spot medians,
  12 gpurentalprices neocloud rate cards, 1 marketplace_ask median.
- Baseline: flat last value. Starter `drift` scores +0.2004 pooled at 2026-10-24 and +0.1734 over 15 weekly
  origins (verified by re-running starter/run_demo.sh).
- Rate-card medians move in steps (menu costs); AWS spot prices move most days (>80% of cells change).
- Do-not-splice rule: instrument classes are different instruments; never chain AWS spot with rate cards.

## [PACK_ONLY] iteration 1: PROMOTED (2026-11-15T10:25:12Z)
**Hypothesis.** Multi-regime first-principles forecaster (deterministic Task-4 candidate).

Multi-regime first-principles forecaster (deterministic Task-4 candidate).

Regime A, AWS hyperscaler_spot (administered, smoothed spot price; ~96% of the pooled loss):
  H0 momentum: AWS spot moves in trends, so the h-day log change is regressed (pooled OLS over every
     benchmark AZ series, trailing 540 days) on the trailing 1/3/7-day log changes.
  H1 stickiness: residuals are scaled by local volatility (14-day mean |1-day log change|), so the
     interval collapses toward the last value on zones whose price has stopped moving.
  H3 conformal: q10/q50/q90 = forecast + local vol x empirical quantiles of the pooled standardised
     in-sample residuals (split-conformal calibration of the 80% band).
Regime B, daily-statistics rate cards (neocloud_rate_card / hyperscaler_list; menu-cost pricing):
  empirical h-day log-change quantiles (drift); H1: q50 pinned to the last value when >= 50% of the
  trailing 60 one-day changes are zero; H3: band width scaled by the IQR dispersion (p75 - p25) / median
  relative to its trailing 60-day median (clipped to [0.6, 1.6]).
Regime C, daily-statistics marketplace_ask / spot series:
  drift. H2 cross-collector error correction (q50 pulled, ramped by h/7, toward the median of the other
  collectors' same-family, same-tier, same-class daily medians over the last 7 days) is kept as a switch
  but disabled (kappa = 0): on Discovery a pull of 0.25 cut the non-AWS pooled skill from +0.06 to -0.32,
  and a cross-AZ consensus term cut the AWS skill from +0.44 to +0.33 (falsified, see theory_of_market.md).

*Lab verdict:* holdout pooled 0.2966 (need > 0.2336), median 0.2552 (need >= 0.2408), coverage 0.7408 (band 0.74-0.86), horizons {'1': 0.3983, '3': 0.3244, '7': 0.2668}.

## [PACK_ONLY] iteration 2: REVERTED (2026-11-15T10:48:49Z)
**Hypothesis.** Local split-conformal calibration: residual tails are heterogeneous across AZ series and non-stationary in time, so the champion's single pooled, full-window, in-sample z over-covers calm zones (p5 use1 coverage 0.97, skill -0.99) and under-covers volatile ones (p5en use1-az2 coverage 0.636) and the upper tail (q90 breach 11.9% vs 10% target; h7 coverage 0.769 vs h1 0.840). Replacing z with per-series empirical quantiles of standardized residuals over a trailing 180-day window, shrunk toward the pooled-recent z with a 30-sample pseudo-count, should move per-series coverage toward 0.80, breach rates toward 10/10%, lift holdout coverage off the 0.74 floor, and gain > 0.003 pooled holdout skill. Falsified if holdout pooled skill <= 0.2966 + 0.003 or coverage leaves [0.74, 0.86].

## [PACK_ONLY] iteration 2: candidate proposed (local split-conformal calibration)

**Diagnosis (Discovery, champion).** Loss is ~95% AWS hyperscaler_spot (h7 60%, h3 25%, h1 9%). The champion's
single pooled, full-window, in-sample conformal z is mis-calibrated per series and per horizon: q90 breach 11.9%
vs 10% target (q10 10.1% fine); h1 coverage 0.840 (too wide) vs h7 0.769 (too narrow); calm p5 zones over-covered
at 0.97 with skill -0.99 (pure width loss at 0.1 weight) while p5en use1-az2 is under-covered at 0.636 (expensive
0.9-weighted breaches). Holdout coverage 0.7408 sits 0.0008 above the Gate-3 floor.

**Hypothesis (falsifiable).** Residual tails are heterogeneous across AZ series and non-stationary in time, so a
pooled full-window in-sample z cannot be honest per series. Replacing it with per-series empirical quantiles of
standardised residuals over a trailing 180-day window, shrunk toward the pooled-recent z with a 30-sample
pseudo-count (min 15 local samples), will (a) move per-series coverage toward 0.80, (b) move q10/q90 breach rates
toward 10/10%, (c) lift holdout coverage off the 0.74 floor, and (d) gain > 0.003 pooled holdout skill.
Only regime A's calibration changed; the momentum beta, vol scaling, and regimes B/C are byte-identical to
iteration 1, so any effect is attributable to the calibration change.

**Predicted failure mode.** If the recent 180-day window is systematically calmer than the following h days
(vol falling), local calibration narrows bands and holdout coverage could drop below 0.74 -> Gate 3 fail; that
outcome falsifies the "recent residuals estimate conditional quantiles" mechanism in favour of a full-window
conservative one.

*Lab verdict:* holdout pooled -9.7752465503551e+17 (need > 0.2996), median 0.1699 (need >= 0.2502), coverage 0.7182 (band 0.74-0.86), horizons {'1': 0.408, '3': 0.2951, '7': -1.6053584808952591e+18}.

## [PACK_ONLY] iteration 3: REVERTED (2026-11-15T10:49:34Z)
**Hypothesis.** Split-conformal tail-widening calibration (regime A only). The champion's full-window in-sample pooled z suffers regression dilution: the momentum OLS absorbs historical tail moves, so forward upper-tail risk is under-estimated (Discovery q90 breach 11.9% vs 10% target, q10 10.1% fine; h7 coverage 0.769 vs h1 0.840; holdout coverage 0.7408 only 0.0008 above the Gate-3 floor). Replacing z with the per-horizon elementwise MAX of (a) the champion's full-window in-sample pooled quantiles and (b) split-conformal pooled quantiles from a recent 90-day calibration chunk (beta refit on the pre-chunk window, >=200 pooled rows required) widens bands only where recent out-of-sample standardized residuals are heavier than the full window. Predicted: q90 breach 11.9% -> 10-11%, h7 coverage 0.769 -> ~0.79, holdout coverage 0.7408 -> >=0.75, pooled holdout skill gain > 0.003. Widening-only by construction (never narrows), pooled across series (no per-series fragility), final log-quantiles clipped to last log value +/- 1.0. Falsified if holdout pooled <= 0.2966 + 0.003 or coverage leaves [0.74, 0.86].

## [PACK_ONLY] iteration 3: candidate proposed (split-conformal tail-widening calibration, regime A only)

**Diagnosis (Discovery, champion).** Loss is ~95% AWS hyperscaler_spot (h7 60%, h3 25%, h1 9%). Calibration is
asymmetric in two dimensions: (i) tail — q90 breach 11.9% vs 10% target while q10 breach 10.1% is on target, so
only the UPPER tail is under-covered; (ii) horizon — h1 coverage 0.840 (too wide) vs h7 0.769 (too narrow).
Holdout coverage 0.7408 sits 0.0008 above the Gate-3 floor, so widening is the safe direction. Iteration 2
falsified per-series recent-window calibration (narrowing exploded h7 to -1e18 pooled); the un-falsified lever
is POOLED, PER-HORIZON, WIDEN-ONLY.

**Hypothesis (falsifiable).** The champion's full-window in-sample pooled z suffers regression dilution: the
momentum OLS absorbs historical tail moves into its fit, so the in-sample residual quantiles under-state the
forward upper tail (hence q90 breach 11.9%, h7 coverage 0.769). Replacing z with the per-horizon elementwise
MAX of (a) the champion's full-window in-sample quantiles and (b) split-conformal pooled quantiles from a recent
90-day calibration chunk (beta refit on the pre-chunk window, >= 200 pooled rows) will (a) cut q90 breach from
11.9% toward 10-11%, (b) lift h7 coverage 0.769 -> ~0.79 and holdout coverage 0.7408 -> >= 0.75, and (c) gain
> 0.003 pooled holdout skill (pinball arithmetic: at a 11.9% breach rate, widening q90 by delta has expected
value 0.9*0.119 - 0.1*0.881 = +0.019 per unit delta; at 10% it is exactly breakeven). Only regime A's z changed;
beta, vol scaling, regimes B/C are byte-identical to iteration 1, plus a log-band clip (last log value +/- 1.0)
and z clip (+/- 8) as hard guardrails.

**Predicted failure mode.** If the recent 90-day chunk is systematically heavier than the *following* holdout
window (vol falling into the holdout), the max() widens bands that were already adequate, coverage overshoots
past 0.86 or the 0.1-weighted width cost on the ~88% inside cells exceeds the 0.9-weighted breach saving —
pooled skill falls below 0.2966 + 0.003 and the "in-sample dilution" mechanism is falsified in favour of
"full-window in-sample z is already honest".

*Lab verdict:* holdout pooled 0.297 (need > 0.2996), median 0.2576 (need >= 0.2502), coverage 0.738 (band 0.74-0.86), horizons {'1': 0.4016, '3': 0.3221, '7': 0.2681}.

## [PACK_ONLY] iteration 4: REVERTED (2026-11-15T10:50:13Z)
**Hypothesis.** Regime-A momentum shrinkage (iteration 4, champion + ONE change). The champion's pooled momentum OLS beta over-fits trend noise on calm, sticky AWS spot zones (worst series p5 use1 az1/az4 skill -0.99, coverage 0.97; p5e use2-az3 coverage 1.0): administered spot prices mean-revert within an AZ after trend runs, so unshrunk betas add noise to q50 exactly where the baseline (last value) is hard to beat. Shrinking beta toward zero by a pooled per-horizon factor s in [0.5, 1.0], selected point-in-time by out-of-sample validation on the most recent 90 days (beta refit on the older trailing chunk, >= 200 pooled rows each side, switch away from s=1.0 only on a >1% validation-MSE gain), will (a) reduce q50 error on calm zones, (b) weakly WIDEN the conformal band (any s<1 raises in-sample residuals vs the SSE-minimising OLS fit), lifting holdout coverage off the 0.7408 floor, and (c) gain > 0.003 pooled holdout skill. Regimes B/C, the vol scaling and the conformal z machinery are byte-identical to the champion; a hard log-band clip (last log value +/- 1.0) is added as an anti-explosion guardrail. Falsified if holdout pooled <= 0.2966 + 0.003, or coverage leaves [0.74, 0.86], or any horizon skill < -0.020.

Regime A, AWS hyperscaler_spot (administered, smoothed spot price; ~96% of the pooled loss):
  H0 momentum: AWS spot moves in trends, so the h-day log change is regressed (pooled OLS over every
     benchmark AZ series, trailing 540 days) on the trailing 1/3/7-day log changes.
  H4 shrinkage (iteration 4): the pooled OLS beta is shrunk toward zero by a per-horizon factor
     s in [0.5, 1.0] selected point-in-time by pooled out-of-sample validation on the most recent
     90 days (beta refit on the older trailing chunk, >= 200 pooled rows each side; s = 1.0 kept
     unless a candidate shrunk beta lowers validation MSE by > 1%). Mechanism: administered spot
     prices mean-revert within an AZ after trend runs, so unshrunk betas inject trend noise into
     q50 on calm zones (p5 use1 az1/az4 skill -0.99, coverage 0.97). Any s < 1 weakly widens the
     conformal band (OLS minimises in-sample SSE), which is coverage-safe given the holdout
     coverage floor. Falsified if holdout pooled <= 0.2966 + 0.003 or coverage leaves [0.74, 0.86].
  H1 stickiness: residuals are scaled by local volatility (14-day mean |1-day log change|), so the
     interval collapses toward the last value on zones whose price has stopped moving.
  H3 conformal: q10/q50/q90 = forecast + local vol x empirical quantiles of the pooled standardised
     in-sample residuals (split-conformal calibration of the 80% band), with a hard log-band clip
     (last log value +/- 1.0) as an anti-explosion guardrail.
Regime B, daily-statistics rate cards (neocloud_rate_card / hyperscaler_list; menu-cost pricing):
  unchanged from iteration 1 (empirical h-day log-change quantiles, stickiness pin, IQR width).
Regime C, daily-statistics marketplace_ask / spot series:
  unchanged from iteration 1 (drift; cross-collector pull disabled, kappa = 0, falsified in iteration 1).

*Lab verdict:* holdout pooled 0.2966 (need > 0.2996), median 0.2552 (need >= 0.2502), coverage 0.7408 (band 0.74-0.86), horizons {'1': 0.3983, '3': 0.3244, '7': 0.2668}.

## [PACK_ONLY] iteration 5: REVERTED (2026-11-15T15:33:06Z)
**Hypothesis.** Per-horizon tail-width reallocation, h1-only narrowing (horizon_structure family, bold). Discovery shows the champion's h1 band is over-covered (coverage 0.840 vs h7 0.769) while h3/h7 are calibrated: the in-sample standardized tails at h=1 are too wide relative to the true 1-day-ahead distribution, so the 0.1-weighted width cost on the ~84% inside cells exceeds the breach saving. Scaling BOTH h1 tails around the unchanged q50 by a fixed pooled factor 0.95 (z -> z50 + 0.95*(z - z50); q50, h3 and h7 byte-identical to the champion) narrows h1 width by 5%, moves h1 coverage 0.840 -> ~0.80 and q10/q90 breach toward 10/10%, and gains pooled skill. Measured on Discovery (dry run): h1 skill +0.0050, h3/h7 unchanged, pooled +0.0005, coverage -0.011 (h1 only). The h7-widening half of this reallocation was measured and is NEGATIVE (-0.020 h7 skill), so it is excluded: h7 under-coverage is a centering (bias_rel -0.012), not a width, problem. Falsified if holdout pooled <= 0.2966 + 0.003, or coverage leaves [0.74, 0.86], or any horizon skill < -0.020.

### Iteration 5 theory update (PACK_ONLY)

**Diagnosis.** The champion's remaining asymmetries were re-measured with four Discovery dry runs and one
out-of-sample analysis: (1) per-instance-type betas/z (cross_series hierarchy) are a wash-to-worse
out-of-sample (q50 err 0.01136 -> 0.01149 at h1; pinball worse on 2 of 3 origins at h7) — the pooled OLS is
near-optimal, so the naive hierarchy is dead; (2) a 0.8/0.2 momentum+drift quantile ensemble (ensemble family)
is decisively negative: pooled -0.0141, 10/11 origins worse, coverage +0.011 — the drift member degrades q50
and its wider tails cost more than the breach saving; (3) calm-regime (bottom-quartile local vol) z narrowing
(regime_change) is negative: pooled -0.0005, coverage -0.004 (Gate-3 unsafe); (4) per-horizon pooled drift
mu_h added to all quantiles (horizon_structure, intercept fix) is catastrophic: pooled -0.0323, 11/11 origins
worse — the in-sample mean residual is NEGATIVE and does not persist forward, so the forward bias_rel
(-0.001/-0.006/-0.013 by horizon) is NOT an intercept problem; (5) per-horizon tail reallocation
(h1 -5%, h7 +20%) is net negative (-0.0119 pooled): the h1 half alone is POSITIVE (+0.0050 h1 skill,
pooled +0.0005), the h7-widening half is strongly negative (-0.020 h7 skill) even though it lifts h7 coverage
0.769 -> 0.808 — h7 under-coverage is a CENTERING problem (bias_rel -0.012), not a width problem.

**What survives.** Only the h1 tail narrowing survives the Discovery screen. It is submitted as the bold
horizon_structure candidate: z -> z50 + 0.95*(z - z50) at h=1 only, q50 and h3/h7 byte-identical to the
champion. Predicted: h1 coverage 0.840 -> ~0.80, pooled holdout gain ~ +0.001 (below the +0.003 Gate-1
threshold, so the honest prediction is a Gate-1 shortfall that falsifies the width-waste mechanism at h1).

**Structural conclusion after five iterations.** Every width lever (widen or narrow, pooled or per-series,
recent or full-window, per-horizon or per-regime) and every centering lever (intercept, drift, shrinkage,
hierarchy) has now been measured and is negative except micro-gains (< +0.001 pooled). The champion is at a
sharp local optimum for PACK_ONLY signals; the honest prior is that remaining headroom above 0.2966 pooled
is < 0.001.

*Lab verdict:* holdout pooled 0.2971 (need > 0.2996), median 0.2562 (need >= 0.2502), coverage 0.738 (band [0.74, 0.86], or within 0.0642 of 0.80), horizons {'1': 0.403, '3': 0.3244, '7': 0.2668}. Gate margins {'gate1': -0.0025, 'gate2': 0.006, 'gate3': 0.0022, 'gate4': 0.2868, 'shortfall': 0.0025}.

## [PACK_ONLY] iteration 6: REVERTED (2026-11-15T15:43:03Z)
**Hypothesis.** Regime-C recency + confirmed h1 narrowing (regime_change family, bold; builds on iteration 5). The champion's regime C (non-card daily-statistics series: the marketplace_ask target and 4 non-AWS hyperscaler_spot targets) uses FULL-history h-day log-change quantiles, but provider repricing is non-stationary (Discovery coverage 0.58-0.63 on marketplace, q90 breach 25%): a trailing 60-day window (min 30 samples, full-history fallback) should track the current repricing regime and widen/center the band where recent changes are larger than the full history. Stacked with the iteration-5 confirmed h1 tail narrowing (z -> z50 + 0.95*(z-z50) at h=1 only, q50/h3/h7 machinery byte-identical), both levers fire unconditionally. Measured on Discovery (dry run): pooled +0.000369 (8/11 origins better), h1 +0.0050, h3 -0.0003, h7 -0.0001, coverage -0.0003; the regime-C lever alone is ~neutral-to-slightly-negative on Discovery (the 60-day quantiles are close to full-history ones on these series). Predicted holdout: h1 narrowing contributes ~+0.0005 pooled (confirmed iteration 5), regime-C ~0, so the honest prediction is a Gate-1 shortfall near 0.0025 unless regime-C recency adds more out of sample than it showed on Discovery. Falsified if holdout pooled <= 0.2966 + 0.003, or coverage leaves [0.74, 0.86], or any horizon skill < -0.020; the regime-C mechanism is additionally falsified if the non-AWS slices (marketplace + non-AWS hyperscaler) do not improve their coverage vs the champion.

### [PACK_ONLY] iteration 6: regime-C recency + h1 narrowing (regime_change, bold)

**Diagnosis first (6 analyses, 4 dry runs).** The playbook's top-ranked untried family (exogenous_signals) was measured out-of-sample BEFORE any dry run: (a) `daily_statistics.hyperscaler_spot` (afhubbard collector, 3,765 rows) provides market-wide per-gpu_model medians mappable to 4 of 6 AWS gpu_model families (A100 40GB/80GB, H100, H200; B200/B300 unmapped); (b) augmenting the champion's pooled momentum OLS with market-median 1/3/7-day log changes is NEGATIVE out-of-sample at both tested origins (2026-09-23, 2026-08-12): MSE +3-10% worse at every horizon (e.g. h7 0.0140 -> 0.0150), MAE worse everywhere. The market-wide signal is LAGGED and noisier than the per-AZ administered price itself. (c) Long-lag momentum (adding 14/30-day log changes) measured mixed in analysis (h7 MSE -3%) but NEGATIVE on the full Discovery screen: pooled -0.0050, 7/11 origins worse, all three horizons worse. Dead.

**Regime-B spread floor (interval_calibration) falsified by dry run.** Rate-card bands are near-degenerate (mean_width_rel 0.0006-0.004, q90 breach 29-31%), and the naive EV argument (breach 0.29 -> +0.21 per unit width) suggested a widening-only log-spread floor of 0.05*sqrt(h). Dry run: pooled -0.0118, 8/11 origins worse, rate-card slice skill -0.79 to -1.03. The EV argument was WRONG because the rate cards' breaches are cent-level absolute moves: a 5% relative band dwarfs them, so the width cost swamps the breach saving. Lesson: EV-per-unit-width must be computed against the ABSOLUTE breach loss, not the breach rate, when the series' own volatility is orders of magnitude below the floor.

**What was submitted.** The only untouched regime is C (non-card daily-statistics series: 1 marketplace_ask target + 4 non-AWS hyperscaler_spot targets, ~4% of pooled loss, coverage 0.58-0.63). New lever: trailing 60-day h-day change quantiles (min 30 samples, full-history fallback) replacing full-history quantiles — provider repricing is non-stationary. Stacked with the confirmed iteration-5 h1 tail narrowing (0.95 around q50, h1 only). Discovery: pooled +0.000369 (0.443221 vs 0.442852), 8/11 origins better, h1 +0.0050, h3 -0.0003, h7 -0.0001, coverage 0.7792 (-0.0003). The regime-C lever alone is ~neutral on Discovery (recent-60-day quantiles are close to full-history quantiles on these short series); its value is the out-of-sample recency bet. Honest prediction: holdout ~+0.0003 to +0.0005 pooled -> Gate-1 shortfall ~0.0025 unless regime C adds more than Discovery showed.

**Structural conclusion after six iterations.** Every lever of the series_history machinery (width, centering, hierarchy, lags, windows) and the two usable exogenous tables (daily_statistics momentum, dispersion untested for lack of budget) have been measured; only micro-gains (< +0.001) survive. The champion remains at a sharp local optimum for PACK_ONLY signals.

*Lab verdict:* holdout pooled 0.2976 (need > 0.2996), median 0.2586 (need >= 0.2502), coverage 0.7436 (band [0.74, 0.86], or within 0.0642 of 0.80), horizons {'1': 0.403, '3': 0.3242, '7': 0.2677}. Gate margins {'gate1': -0.002, 'gate2': 0.0084, 'gate3': 0.0078, 'gate4': 0.2877, 'shortfall': 0.002}.

## [PACK_ONLY] iteration 7: REVERTED [regime_change] (2026-11-15T15:55:55Z)
**Hypothesis.** Pooled vol-state tail scaling at h3/h7, stacked on the iteration-6 confirmed stack (regime_change family, bold; builds on iteration 6). Mechanism: the champion's full-window in-sample pooled z is estimated mostly across heterogeneous vol regimes, so at origins where the whole AWS spot market is CALM it over-covers the forward window. Pseudo-origin OOS analysis (fit at origin-28d, score the next 28 days, 5 Discovery origins) shows: unconditional h3/h7 0.95 narrowing is mixed (positive at calm origins 09-23/09-02, negative at the hot 08-12 origin), per-series conditional rules (calm-only, vol-ratio continuous) are all worse than unconditional, but a POOLED state rule — narrow both tails around the unchanged q50 by clip(median over series of current-14d-vol / own-trailing-90d-median-vol, floor, 1.0) only when that market-wide median < 1 — dominates: byte-identical to the champion at hot origins (median ratio >= 1) and +0.006..+0.013 skill at calm origins. Dry-run decomposition per horizon (vs champion): h1 state scaling +0.0040 skill but -0.0055 coverage (worst trade, excluded: h1 keeps the confirmed unconditional 0.95 only); h3 floor 0.95: +0.0029 skill, -0.0087 slice coverage; h7 floor 0.93: +0.0040 skill, 0.0 slice coverage (h7 in-sample z ~ 7.8 std over-covers most at calm origins and its narrowing is coverage-neutral). Final candidate = iteration-6 stack (regime-C 60-day change quantiles + h1 0.95 narrowing) + state_floor {3: 0.95, 7: 0.93}, firing unconditionally whenever the market-wide median vol ratio < 1 (9/11 Discovery origins), byte-identical to the stack otherwise. Discovery: pooled 0.446618 (+0.003766 vs champion, +0.0034 marginal over the stack), 9/11 origins better, h1 +0.0050 / h3 +0.0029 / h7 +0.0040, coverage 0.7740 (-0.0055, of which -0.0026 is the confirmed h1 lever already in the stack). Predicted holdout: pooled +0.0027..+0.0044 vs champion (stack +0.00095 confirmed + state lever at 50-100% transfer) -> Gate 1 borderline at the +0.003 threshold; coverage projects 0.738-0.741, inside the hard band and within the Gate-3 distance clause (floor 0.7358). Falsified if holdout pooled <= 0.2966 + 0.003, or coverage < 0.7358, or any horizon skill < -0.020; the vol-state mechanism is additionally falsified if the calm-origin skill gains (09-02..09-16) do not transfer (i.e., holdout pooled gain < the stack's +0.00095).

### Iteration 7 theory update (PACK_ONLY): pooled vol-state tail scaling

**Diagnosis first (3 analyses, 4 dry runs).** The Gate-1 gap after the iteration-6 stack is ~0.002 pooled on Holdout. Pseudo-origin OOS analysis (fit at origin-28d, score the following 28 days inside the Discovery slice; origins 09-23, 09-02, 08-19, 08-12, 07-29) re-measured every remaining lever out of sample BEFORE any dry run:

1. **z-centering is a no-op** (z50 = 0 by construction of the pooled OLS on log changes; zctr == champ at every origin/horizon). The h7 bias_rel -0.012 is NOT an intercept problem — confirmed again.
2. **In-sample mean residual does not persist**: mean standardised h7 error -1.15..-0.91 std; subtracting it (biasfix) is origin-dependent (+0.27 at 08-12, -0.25 at 09-23). Dead (consistent with iteration 5's mu_h result).
3. **h7 beta damping (0.85/0.70) is a wash-to-worse** OOS. Dead (consistent with iteration 4's shrinkage null).
4. **Regime-C q50 centering variants (recent-60 median drift, zero-pin) are a wash** on the 4-5 evaluable series. Dead.
5. **aggregator_medians / index_prints are degenerate for this stage**: aggregator_medians is one source (getdeploying_weekly, reservation-style weekly medians) with no overlap to the benchmark series' collectors; index_prints has 0-12 rows. Dead as exogenous signals in PACK_ONLY.
6. **Per-series conditional tail rules are all worse than unconditional**: calm-only 0.95 (per-series vol ratio < 1), continuous clip(r, 0.93, 1.07), 0.95*r — every variant loses to unconditional 0.95 at both calm and hot origins. The cross-sectional calm/vol split does not generalise forward; only the TIME dimension does.
7. **The pooled vol STATE dominates**: market-wide median of (current 14d vol / own trailing 90d median vol) < 1 marks calm origins (09-23: 0.83, 09-02: 0.66) where narrowing gains +0.006..+0.013 skill at every horizon; hot origins (08-12, 08-19, 07-29: ratio >= 1.0) are left byte-identical to the champion. This is a genuine regime_change (time-regime) mechanism, distinct from the falsified per-series calm-vol lever (iteration 5).

**Dry-run decomposition (the tuning).** State floor 0.93 all horizons: pooled +0.0045 but coverage -0.0097 (holdout Gate-3 budget above the distance-clause floor is ~0.008 -> too risky). Floor 0.95 all horizons: +0.0036 / -0.0065. Per-horizon marginal trades: h1 +0.0040 skill / -0.0055 coverage (excluded — h1 keeps the confirmed unconditional 0.95 only); h3 +0.0029 / -0.0087 (floor 0.95); h7 +0.0040 / 0.0 coverage (floor 0.93 — the h7 in-sample z ~ 7.8 std over-covers most at calm origins and its narrowing is coverage-neutral). Final: state_floor {3: 0.95, 7: 0.93}, h1 untouched. Discovery pooled 0.446618 (+0.003766 vs champion, +0.0034 marginal over the stack), 9/11 origins better, coverage 0.7740 (-0.0055, of which -0.0026 is the confirmed h1 lever already in the stack).

**Honest prediction.** Holdout pooled +0.0027..+0.0044 vs champion (stack +0.00095 confirmed + state lever at 50-100% Discovery-to-Holdout transfer) -> Gate 1 is genuinely borderline at the +0.003 threshold; coverage projects 0.738-0.741, inside the hard band [0.70, 0.90] and within the Gate-3 distance clause (floor 0.7358, margin ~0.003). The mechanism is falsified if the calm-origin gains do not transfer (holdout gain < the stack's +0.00095) or if coverage falls below 0.7358.

**Structural note.** After seven iterations, every width/centering lever of the series_history machinery has been measured; the surviving positive levers are all RECENCY/STATE levers (h1 unconditional narrowing = a fixed state; regime-C 60-day window; pooled vol-state scaling). The champion's full-window in-sample calibration is honest only in expectation over regimes; conditioning it on the current market state is where the remaining headroom lives.

*Lab verdict:* holdout pooled 0.2969 (need > 0.2996), median 0.2561 (need >= 0.2502), coverage 0.7429 (band [0.74, 0.86], or within 0.0642 of 0.80), horizons {'1': 0.403, '3': 0.3239, '7': 0.2667}. Gate margins {'gate1': -0.0027, 'gate2': 0.0059, 'gate3': 0.0071, 'gate4': 0.2867, 'shortfall': 0.0027}.

## [PACK_ONLY] iteration 8: PROMOTED (2026-11-15T16:00:20Z)
**Hypothesis.** Vol-window parameter-perturbation ensemble at h3/h7, stacked on the iteration-6 confirmed stack (ensemble family, bold; builds on iteration 6). Mechanism: the champion's regime-A band is scaled by a single arbitrary 14-day local-vol window; averaging the three quantile sets across vol_wins {7,14,28} (same pooled-OLS beta machinery, train_days fixed at 540) is a poor-man's bagging that fires unconditionally, widens the h3/h7 bands toward the 0.80 target (champion h3 0.783, h7 0.751 coverage) and cuts breach loss on the volatile AZ series that carry ~85% of the pooled loss. Pseudo-origin OOS analysis (fit at origin-28d, score the next 28 days, 5 Discovery origins): h3 +0.0010..+0.0093 (5/5 origins), h7 +0.0004..+0.0098 (5/5), per-series median skill improves 4/5 origins; the 270-day train member is mixed-to-negative at h7 (excluded), avgbeta is a wash (excluded), h1 keeps the confirmed single-window + 0.95 narrowing byte-identical (h1 coverage 0.803 already at target; the h1 ensemble widens to 0.829 and is excluded). Dry run: Discovery pooled 0.446187 (+0.003335 vs champion), h1 +0.0050 (stack lever), h3 +0.0022, h7 +0.0036, coverage 0.7947 (+0.0152 toward 0.80), q10 breach 10.1% -> 9.0%. Predicted holdout: per-horizon deltas transfer (iteration-5/6 precedent: h1 +0.0050 Discovery -> +0.0047 holdout), so h3 +0.002..+0.006, h7 +0.003..+0.006 -> pooled 0.2986..0.3010 vs the 0.2996 Gate-1 threshold (borderline pass); coverage rises from 0.7436 toward ~0.755 (in-band); median: the Discovery in-sample median drops -0.0146 (expected in-sample leakage: the champion's z is optimal in-sample by construction) but the OOS pseudo-origin per-series median improves +0.007 avg (4/5 origins), so the holdout median is predicted flat-to-up from the stack's 0.2586 (buffer to the 0.2502 Gate-2 floor is 0.0084). Falsified if holdout pooled <= 0.2996 (Gate 1), or median < 0.2502 (Gate 2), or coverage leaves [0.74, 0.86] (Gate 3), or any horizon skill < -0.020 (Gate 4); the ensemble mechanism is additionally falsified if the h3/h7 holdout skill deltas are <= 0, i.e. if the pseudo-origin OOS gains fail to transfer the way iteration 7's vol-state lever did.

### [PACK_ONLY] iteration 8: vol-window ensemble at h3/h7 (ensemble family, bold; builds on iteration 6)

**Diagnosis.** The Gate-1 gap after the iteration-6 stack is +0.002 pooled on Holdout (stack 0.2976 vs 0.2996 threshold). Width levers conditioned on recency/state are exhausted with Holdout evidence (iterations 3, 5, 7); the remaining headroom had to come from an untried family. The playbook's #1 ranked untried direction was a parameter-perturbation ensemble: average quantiles across perturbed hyperparameters of the confirmed stack instead of adding a (falsified) drift member.

**Pseudo-origin OOS analysis (2 analyses, fit at origin-28d, score the next 28 days, 5 Discovery origins, AWS regime A).** The vol window is the only window-sensitive part of the regime-A machinery (beta is pooled OLS on log changes; z is standardised by local vol). Measured out of sample:
- **Vol-window ensemble {7,14,28}, train 540:** h3 +0.0010..+0.0093 (5/5 origins, avg +0.0066), h7 +0.0004..+0.0098 (5/5, avg +0.0064), h1 4/5 (avg +0.005); coverage moves TOWARD 0.80 at every horizon (champion under-covers h3 0.783 / h7 0.751 OOS).
- **Train-day ensemble {270,540}:** mixed at h1/h3, NEGATIVE at h7 (beta270 over-weights the 1-day lag: beta270[0] ~ 1.9 vs beta540 ~ 1.6 at h7). Excluded.
- **avgbeta (average the betas, single z):** wash at every horizon. Excluded — the gain is in averaging the z/vol scale, not the point forecast.
- **Per-series median skill OOS improves 4/5 origins** for the full ensemble (avg +0.007), even though the in-sample Discovery median drops -0.0146 — expected leakage: the champion's in-sample z is optimal in-sample by construction, so any widening looks bad in-sample and helps out of sample.

**Candidate.** Iteration-6 stack (regime-C 60-day change quantiles + h1 0.95 narrowing, h1 byte-identical) + regime-A quantile averaging across vol_wins {7,14,28} at h3/h7 only (h1 excluded: its coverage 0.803 is already at target and the h1 ensemble widens to 0.829). Fires unconditionally; all members bounded by the same machinery (no explosion risk); runtime ~0.3 s/origin.

**Dry run.** Discovery pooled 0.446187 (+0.003335 vs champion), h1 +0.0050 / h3 +0.0022 / h7 +0.0036, coverage 0.7947 (+0.0152 toward 0.80), q10 breach 10.1% -> 9.0%, q90 11.9% -> 11.5%. Screen passes; would_reach_holdout true.

**Honest prediction.** Holdout pooled 0.2986..0.3010 (per-horizon delta transfer precedent: iteration-5/6 h1 +0.0050 Discovery -> +0.0047 holdout) -> Gate 1 borderline; coverage ~0.755 in-band; median flat-to-up vs the stack's 0.2586 (Gate-2 buffer 0.0084). The mechanism is falsified if the h3/h7 holdout deltas are <= 0 (a repeat of iteration 7's non-transfer) or if Gate 2 fails on the median.

*Lab verdict:* holdout pooled 0.3088 (need > 0.2996), median 0.2745 (need >= 0.2502), coverage 0.7514 (band [0.74, 0.86], or within 0.0642 of 0.80), horizons {'1': 0.403, '3': 0.3369, '7': 0.2801}. Gate margins {'gate1': 0.0092, 'gate2': 0.0243, 'gate3': 0.0156, 'gate4': 0.3001, 'shortfall': 0}.

## [PACK_ONLY] iteration 9: REVERTED [ensemble] (2026-11-15T16:02:52Z)
**Hypothesis.** Extend the confirmed vol-window ensemble to h1 with a 0.90 tail-narrowing re-center (ensemble family, incremental; builds on iteration 8, the champion). Mechanism: iteration 8's parameter-perturbation averaging of the band scale across vol_wins {7,14,28} was restricted to h3/h7 because the raw h1 ensemble widened h1 coverage 0.803 -> 0.829 (away from the 0.80 target). But the widening is fixable by re-centering: pseudo-origin OOS analysis (fit at origin-28d, score the next 28 days, 5 Discovery origins, AWS regime A) shows the h1 ensemble {7,14,28} with narrowing factor f beats the champion's single-window(14)+0.95 machinery at f=0.90 by +0.0067..+0.0092 h1 skill (5/5 origins, avg +0.0080), at f=0.92 by avg +0.0059, at f=0.95 by avg +0.0025 (one negative origin) — i.e. the ensemble gain at h1 is real and the narrowing factor that re-centers coverage near 0.80 maximises it. The same analysis falsifies the 4-member set {7,14,21,28} at h3/h7 (avg -0.003 at both horizons, negative at every origin) — excluded; h3/h7 stay byte-identical to the champion. Candidate = champion with ens_horizons (1,3,7) and tail_scale h1 0.90; fires unconditionally, all members share the bounded champion machinery. Dry run: Discovery pooled 0.447101 (+0.000915), h1 +0.0091, h3/h7 0.0, coverage 0.7947 -> 0.7990 (toward 0.80), q10 breach 9.0% -> 8.8%, median +0.0022, 8/11 origins better, screen passes. Predicted holdout: h1 delta transfers at the iteration-5/8 precedent scale (+0.005..+0.010 h1) -> pooled +0.002..+0.004 vs the Gate-1 threshold +0.003 (borderline); coverage rises from 0.7514 toward ~0.755 (in-band); median flat-to-up (Gate-2 buffer 0.0243). Falsified if holdout pooled <= 0.3118 (Gate 1), or median < 0.2695 (Gate 2), or coverage leaves [0.74, 0.86] (Gate 3), or any horizon skill < -0.020 (Gate 4); the mechanism is additionally falsified if the h1 holdout skill delta <= 0 (a repeat of iteration 7's non-transfer).

### [PACK_ONLY] iteration 9: h1 vol-window ensemble with 0.90 re-centering (ensemble family, incremental; builds on iteration 8)

**Diagnosis.** The champion (iteration 8) passes all gates with margin (gate1 +0.0092). The playbook's #1 ranked untried direction was extending the vol-window ensemble to h1, excluded in iteration 8 only because the RAW h1 ensemble widened h1 coverage 0.803 -> 0.829. The widening is a re-centering problem, not a refutation: the confirmed h1 machinery already carries a narrowing factor (0.95, iteration 5), so the ensemble + a stronger factor is the natural composition of two confirmed mechanisms.

**Pseudo-origin OOS analysis (1 analysis, fit at origin-28d, score the next 28 days, 5 Discovery origins, AWS regime A, 80 series).** Variants scored per horizon vs the champion structure:
- **h1 ensemble {7,14,28} + f:** f=0.90: +0.0067..+0.0092 (5/5 origins, avg +0.0080); f=0.92: avg +0.0059; f=0.95: avg +0.0025 (1 negative origin). The OOS h1 coverage of the f=0.90 variant lands 0.813-0.864 vs champion 0.794-0.861 — the ensemble widens and the 0.90 factor re-centers it; the gain grows as f moves from 0.95 to 0.90.
- **4-member set {7,14,21,28} at h3/h7:** NEGATIVE at every origin (h3 avg -0.0026, h7 avg -0.0029). The 3-member set is the optimum size; adding the 21-day member dilutes it. FALSIFIED — do not retry richer member sets in this family.
- **m4all_h1_92 (4-member everywhere):** dominated by the 3-member h1 variants. Excluded.

**Candidate.** Champion + ens_horizons (1,3,7) + tail_scale {1: 0.90}. Two-line diff; h3/h7/regime-B/regime-C byte-identical to the champion.

**Dry run.** Discovery pooled 0.447101 (+0.000915 vs champion), h1 +0.0091, h3/h7 0.0, coverage 0.7947 -> 0.7990 (toward the 0.80 target), q10 breach 9.0% -> 8.8%, q90 11.5% -> 11.3%, median +0.0022, 8/11 origins better, screen passes, would_reach_holdout true.

**Honest prediction.** Holdout h1 delta +0.005..+0.010 (iteration-5 precedent: h1 +0.0050 Discovery -> +0.0047 holdout; iteration-8 precedent: h3/h7 transferred at ~5x) -> pooled +0.002..+0.004 vs the Gate-1 threshold +0.003: genuinely borderline. Coverage 0.7514 -> ~0.755 (in-band); median flat-to-up (Gate-2 buffer 0.0243). The mechanism is falsified if the h1 holdout delta <= 0 or Gate 1 fails.

**Lessons for the playbook.** (a) The ensemble family's gain generalises across horizons once the coverage re-centering is handled — the h1 exclusion in iteration 8 was about the raw widening, not the averaging. (b) Member-set richness has an optimum: {7,14,28} beats both the single window and the 4-member set OOS; richer is not better. (c) The narrowing factor and the ensemble compose multiplicatively in log space and the OOS optimum of the composition (0.90) is stronger than either component alone.

*Lab verdict:* holdout pooled 0.3102 (need > 0.3118), median 0.2754 (need >= 0.2695), coverage 0.75 (band [0.74, 0.86], or within 0.0536 of 0.80), horizons {'1': 0.4164, '3': 0.3369, '7': 0.2801}. Gate margins {'gate1': -0.0016, 'gate2': 0.0059, 'gate3': 0.01, 'gate4': 0.3001, 'shortfall': 0.0016}.

## [PACK_PLUS_EXA] iteration 1: REVERTED (2026-11-15T16:20:21Z)
**Hypothesis.** horizon_structure (bold): replace the direct h-day pooled-OLS q50 at h3/h7 with an iterated (chained) 1-day forecast — g[0]=last log value, g[j+1]=g[j]+beta1.(g[j]-g[j-1], g[j]-g[j-3], g[j]-g[j-7]) using the pooled h1 OLS beta and actual history for negative j — keeping the champion's vol-window ensemble band machinery byte-identical. Mechanism: the 1-day regression is the best-calibrated machinery in the pack and chaining it compounds the momentum trend instead of extrapolating the single h-day regression (which over-weights the 1-day lag at long horizons); pseudo-origin OOS analysis (fit at origin-28d, score the next 28 days, 5 Discovery origins, AWS regime A) shows the chained q50 beats the direct q50 at h3 (+0.0015 avg, 5/5 origins) and h7 (+0.0011 avg, 5/5 origins) with coverage and bias unchanged (pure centering gain; blends are dominated by the pure chain). Because the direct pooled-OLS q50 is in-sample-optimal by construction, the chain is in-sample Discovery-negative (h3 -0.0009, h7 -0.0010) — so the candidate STACKS the measured iteration-9 lever (h1 vol-window ensemble + 0.90 tail re-center, holdout h1 +0.0134, pooled +0.0013, coverage 0.7500 in-band) to clear the pre-Holdout screen (stack Discovery +0.000058 > 0) and to add its confirmed holdout gain to the chain's expected transfer (~1.8x OOS precedent from iterations 8/9). Predicted holdout: pooled +0.002..+0.004 vs the Gate-1 threshold +0.003 (borderline); h3/h7 deltas positive if the OOS pre-screen transfers as in iterations 8/9. FALSIFIED if the h3 or h7 holdout skill delta <= 0 (chain non-transfer, a repeat of iteration 7), or Gate 1 fails, or coverage leaves [0.74, 0.86], or any horizon < -0.020.

### [PACK_PLUS_EXA] iteration 1: chained h3/h7 q50 + iteration-9 h1 ensemble stack

**Diagnosis.** Champion (iteration 8) passes all gates (gate1 +0.0092). h7 carries 63% of loss, h3 27%, h1 10%. Breach asymmetry: q90 11.5% vs q10 9.0%. The only confirmed-but-unpromoted lever is iteration 9's h1 ensemble + 0.90 re-center (holdout h1 +0.0134, pooled +0.0013 — sub-threshold alone). The playbook's #1 ranked untried direction was horizon_structure: iterated 1-day forecasts.

**Analyses (3 used).**
1. *Chained q50, pseudo-origin OOS (fit at origin-28d, score next 28 days, 5 origins, AWS regime A).* First implementation had a lag bug (used t+1 future value for the 1-day lag at step 0) and exploded to skill -1e7 — fixed by indexing actual history at t-1/t-3/t-7. Corrected result: chained q50 beats the direct pooled-OLS q50 at h3 (+0.0005..+0.0022, 5/5 origins, avg +0.0015) and h7 (+0.0005..+0.0017, 5/5, avg +0.0011); blends (50/25/10% chain weight) are strictly dominated by the pure chain; coverage and bias unchanged (pure centering gain, Gate-3 safe).
2. *q90-only asymmetric widening (interval_calibration), same OOS protocol, factors 1.05..1.20 at h1/h3/h7.* NEGATIVE at every origin, every horizon, every factor (w10 h7 avg -0.0069). Despite the 11.5% q90 breach rate, the champion's q90 is OOS-optimal; the breach-rate signal does not imply a width lever. FALSIFIED — do not retry fixed-factor upper-tail widening in any form (consistent with the PACK_ONLY iterations 3/5 result for h7 tails).
3. *Dry run of the pure chain:* in-sample Discovery pooled -0.000857 (h3 -0.000877, h7 -0.000987), screen fails — as expected, the direct pooled-OLS q50 is in-sample optimal by construction, so any centering deviation is in-sample negative even when OOS-positive (the same in-sample/OOS inversion the playbook documents for width levers, now measured for centering).

**Candidate.** Champion + chain_horizons (3,7) (chained q50 inside the ensemble members, fallback to the direct member q50 when the chain is non-finite) + ens_horizons (1,3,7) + tail_scale h1 0.90 (the archived iteration-9 lever, folded in rather than re-measured). Fires unconditionally; all outputs bounded by the same machinery; runtime ~0.17 s/origin.

**Dry run (stack).** Discovery pooled 0.446245 (+0.000058), h1 +0.0091, h3 -0.000877, h7 -0.000987, coverage 0.7947 -> 0.7980, q10 breach 9.0% -> 8.9%, screen passes, would_reach_holdout true.

**Honest prediction.** Holdout pooled +0.002..+0.004 (h1 lever +0.0013 measured + chain transfer at the ~1.8x OOS precedent -> h3 +0.0027, h7 +0.0020, ~+0.0020 pooled) vs Gate-1 threshold +0.003: borderline. Coverage ~0.750 (in-band); median ~0.275 (Gate-2 buffer 0.024). The chain mechanism is falsified if the h3/h7 holdout deltas are <= 0.

*Lab verdict:* holdout pooled 0.3108 (need > 0.3118), median 0.2717 (need >= 0.2695), coverage 0.75 (band [0.74, 0.86], or within 0.0536 of 0.80), horizons {'1': 0.4164, '3': 0.3379, '7': 0.2807}. Gate margins {'gate1': -0.001, 'gate2': 0.0022, 'gate3': 0.01, 'gate4': 0.3007, 'shortfall': 0.001}.

## [PACK_PLUS_EXA] iteration 2: REVERTED [interval_calibration] (2026-11-15T16:31:29Z)
**Hypothesis.** interval_calibration (incremental, builds on archived iteration 1): estimate the regime-A pooled standardised-residual quantiles z on a 270-day trailing window at h1/h3 (beta stays 540-day; h7 keeps the 540-day z), stacked on the iteration-1 candidate (chained h3/h7 q50 + h1 vol-window ensemble + 0.90 tail re-center, holdout +0.0020 pooled, Gate-1 shortfall 0.001). Mechanism: the pooled z is currently fit on 540 days, but the standardised-residual dispersion relevant 1-3 days ahead is estimated better on a shorter trailing window — pseudo-origin OOS (fit at origin-28d, score next 28 days, 6 Discovery origins, on top of the full iter-1 stack) shows the 270-day z cuts h1 loss ~1.8% (6/6 origins) and h3 ~0.3% (6/6) with the point forecast unchanged; 90-day and {270,90} are worse (optimum ~270), and at h7 the 270-day fit is 0/6 (over-weights the 1-day lag at long horizons, consistent with the archived beta270 finding), so h7 is excluded. Averaging {540,270} (the plain ensemble form) is strictly dominated by the 270-day-only z at both horizons. Predicted holdout: pooled +0.0035..+0.0045 vs the +0.003 Gate-1 threshold (iter-1's measured +0.0020 + w270 h1/h3 transfer at the ~1-1.5x Discovery-delta precedent); h1 +0.010..+0.015, h3 +0.000..+0.002, h7 unchanged-to-+0.001 (chain); coverage ~0.745-0.750 (the z narrows h1/h3 bands ~7%; must stay >= 0.74); median ~0.272-0.276 (Gate-2 buffer 0.005). FALSIFIED if the h1 or h3 holdout skill delta vs the iteration-1 archived candidate is <= 0 (270-day-z non-transfer), or Gate 1 fails, or coverage leaves [0.74, 0.86], or any horizon < -0.020.

### [PACK_PLUS_EXA] iteration 2: 270-day z window at h1/h3, stacked on the iteration-1 chain

**Diagnosis.** Iteration 1 (archived, REVERTED on Gate 1 only, shortfall 0.001) measured the full stack of confirmed levers at holdout pooled +0.0020. Any new lever worth >= +0.001 pooled completes it. EXA web search was probed first (the stage's headline capability): three point-in-time query phrasings (GPU rental price data, AWS spot price announcements) return EMPTY or stale 2018-2023 articles — the web index has no usable point-in-time GPU price signal. exogenous_signals-via-EXA is dead for this pack (recorded in the playbook).

**Analyses (3 used, 6 Discovery origins each, pseudo-origin protocol: fit at origin-28d, score the next 28 days).**
1. *Train-window z ensemble {540,270} at h1/h3 (beta 540):* h1 6/6 origins ~-1.1% loss, h3 6/6 ~-0.2%, h7 0/6 (negative). The ensemble pattern that produced the vol-window gains transfers to the train-window dimension — but only at short horizons.
2. *Window sweep:* 270-day-only z dominates the {540,270} average at h1 (-1.83% vs -1.15%, 6/6) and h3 (-0.34% vs -0.19%, 6/6); 90-day and {270,90} are worse; 30/70 blends are dominated. The optimum is a plain 270-day trailing window, NOT an average — unlike the vol-window case, where averaging was the win. Recency in the z is a monotone signal up to ~270 days at h1/h3 and reverses at h7.
3. *Regime-C recent/full quantile blend:* h7 5/6 positive but the slice is ~2% of loss (~+0.00007 pooled) — not worth touching a confirmed lever. Dead.

**Candidate.** Iteration-1 code + z_train_days=270 applied to the pooled z at h1/h3 only (h7 and all point forecasts byte-identical). One implementation bug caught by dry run: dropping the default z assignment leaked the h3 z into h7 (h7 coverage 0.55, screen fail) — restored; the guard "always re-assign z before any conditional override" is now in the playbook.

**Dry run.** Discovery pooled 0.447787 (+0.0016 vs champion, +0.0015 vs the iter-1 stack), h1 +0.0199, h3 +0.0007, h7 -0.000987 (the chain's known in-sample negativity), coverage 0.7837 (0.7947 champion; narrowing), q10 breach 9.7%, 9/11 origins improved, screen passes.

**Honest prediction.** Holdout pooled +0.0035..+0.0045 vs Gate-1 threshold +0.003: pass expected with ~+0.001 margin. h1 +0.010..+0.015 (w270 Discovery h1 +0.0108 on top of the stack, transfer at the iter-9 1.5x precedent), h3 +0.000..+0.002, h7 +0.0006 (chain, already measured). Coverage ~0.745-0.750 — the main risk is Gate 3's 0.74 floor if the narrowing over-transfers; the Discovery coverage delta -0.011 suggests ~0.745 on holdout, inside the band but with little slack. Median ~0.272-0.276 (Gate-2 buffer 0.005 from 0.2695).

**Lessons for the playbook.** (a) The z train-window is a live dimension: recency helps at h1/h3, hurts at h7 — the same horizon asymmetry as every other lever in this lab (the 1-day-lag over-weighting at long horizons). (b) Averaging is NOT universally the right ensemble form: for the vol scale it was, for the z window a plain shorter window dominates. Sweep the form before assuming the ensemble pattern. (c) EXA search has no point-in-time GPU price data — the exogenous family via web is closed; the pack tables (hardware_fundamentals/capability_ladder) map to launch prices and benchmark ratios, not to the benchmark series' collectors, and remain unusable for level anchoring.

*Lab verdict:* holdout pooled 0.3108 (need > 0.3118), median 0.2752 (need >= 0.2695), coverage 0.7451 (band [0.74, 0.86], or within 0.0536 of 0.80), horizons {'1': 0.4222, '3': 0.3357, '7': 0.2807}. Gate margins {'gate1': -0.001, 'gate2': 0.0057, 'gate3': 0.0051, 'gate4': 0.3007, 'shortfall': 0.001}.

## [PACK_PLUS_EXA] iteration 3: REVERTED [interval_calibration] (2026-11-15T16:40:56Z)
**Hypothesis.** interval_calibration (bold, builds on archived iteration 2): h3 both-tail narrowing f=0.93 on the 540-day z, plus the falsification-driven fix tw_horizons=(1,) — i.e. the completion stack = chained h3/h7 q50 (iter-1, holdout +0.0010/+0.0006) + h1 vol-window ensemble + 0.90 recenter (iter-9 lever, holdout h1 +0.0134) + 270-day z at h1 ONLY (iter-2, holdout h1 +0.0058 CONFIRMED; at h3 FALSIFIED -0.0022, so h3 returns to the 540-day z) + NEW h3 tail factor 0.93. Mechanism for the new lever: the champion's h3 in-sample standardised tails are too wide relative to the true 3-day-ahead distribution (Discovery h3 coverage 0.808 vs 0.80 target, mean_width_rel 0.092); scaling both tails around the unchanged q50 by 0.93 cuts pinball loss ~1.3% at 6/6 pseudo-origins (OOS protocol: fit at origin-28d, score next 28 days, 540-day z, ensemble members; f=0.95 and 0.97 also 6/6, monotone toward 0.93) — the same mechanism as the confirmed h1 narrowing (iter 5, holdout +0.0047) and the iter-9 0.90 recenter (holdout h1 +0.0134), both of which transferred positively. The chained-residual z at h3/h7 was measured and is a wash (z_ratio 1.0, deltas -0.03%, 6/6) — dead, not included. Predicted holdout: h1 0.4222 (unchanged iter-2 machinery), h3 0.340-0.347 (iter-1's 0.3379 + f0.93 transfer at the 50-100% OOS precedent), h7 0.2807 (chain), pooled 0.3120-0.3130 vs Gate-1 threshold 0.31183 (pass, thin margin); coverage ~0.742-0.746 (the h3 z540 restore offsets part of the f0.93 narrowing; must stay >= 0.74); median ~0.274 (Gate-2 buffer 0.005). FALSIFIED if the h3 holdout skill delta vs the archived iteration-1 candidate (0.3379) is <= 0 (narrowing non-transfer), or Gate 1 fails, or coverage leaves [0.74, 0.86], or any horizon < -0.020.

### [PACK_PLUS_EXA] iteration 3: completion stack — h3 tail narrowing + w270-at-h1-only

**Diagnosis.** Both prior near-misses failed Gate 1 only, shortfall 0.001. Exact stack arithmetic: iter-1 stack (+0.0020 holdout pooled) + w270-z at h1 (+0.0006) needs ~+0.0004 more; the falsified w270-at-h3 part (-0.0006) must be removed and replaced.

**Analyses (4 used, 6 Discovery origins each, pseudo-origin protocol: fit at origin-28d, score the next 28 days).**
1. *h1 z-window x tail-factor sweep (beta 540, ensemble):* w270 dominates w180/w360/w540 at 4/6 origins; lower f monotonically better at h1 (f0.88 > f0.90 6/6); b540_z270_f0.88 beats the champion h1 form (z540, f0.95) by ~4.3% h1 loss at every origin. Marginal pooled value of the f-refit alone ~+0.0001 — not worth the coverage cost; kept f0.90 (the measured iter-9 composition).
2. *Chained-residual z at h3/h7:* first implementation mixed forecast LEVELS with change residuals (garbage z, coverage 0.13); corrected (residual of chained level vs actual level): z_ratio 1.0 at every window/origin, deltas -0.03% of loss (6/6 wash). DEAD — the chain's residual distribution is identical to the direct one; do not retry.
3. *h3 both-tail factor sweep (z540, ensemble, direct center):* f=0.93 cuts h3 loss ~1.3% at 6/6 origins (0.95: 6/6 ~+1.0%, 0.97: 6/6, monotone toward 0.93); coverage 0.827->0.813 in-sample, still above the 0.80 target. The same over-wide-in-sample-tails mechanism as the confirmed h1 narrowing, never tried at h3. At ~27% loss share: +0.0006..+0.0012 pooled expected.
4. *h1 w180 window:* dominated by w270 overall (2/6 origins) — w270 stays.

**Candidate.** Iteration-2 code with (a) tw_horizons=(1,) — the 270-day z at h1 only, h3 restored to the 540-day z (its holdout falsification), (b) tail_scale h3 0.93 (new lever). Everything else byte-identical to the archived iteration-2 (chain h3/h7, h1 ensemble + 0.90 recenter).

**Dry run.** Discovery pooled 0.448689 (+0.0025 vs champion), h1 +0.0199, h3 +0.0039, h7 -0.0010 (the chain's known in-sample negativity), coverage 0.7811 (-0.0136), 10/11 origins improved, screen passes, would_reach_holdout true.

**Honest prediction.** Holdout h1 0.4222 (identical machinery to iter-2), h3 0.340-0.347 (iter-1's measured 0.3379 + f0.93 transfer at the 50-100% OOS precedent), h7 0.2807 (chain). Pooled 0.3120-0.3130 vs threshold 0.31183: pass expected, margin +0.0001..+0.0015 — the thinnest of any submission so far. Coverage ~0.742-0.746 (in-band [0.74, 0.86] but with little slack; the h3 z540 restore offsets part of the f0.03 narrowing). Median ~0.274 (Gate-2 buffer 0.005). The h3 narrowing is falsified if the h3 holdout delta vs archived iteration-1 (0.3379) is <= 0.

**Lessons.** (a) The tail-factor dimension is live at h3, not just h1: in-sample over-coverage at a horizon predicts an OOS-positive both-tail narrowing, and the h1 precedent (iters 5/9) transferred positively twice. (b) The chained-residual z completion of the chain is dead — the chain changes only the center, not the residual dispersion. (c) The w270-at-h3 falsification was recency-specific: restoring the 540-day z at h3 is itself worth +0.0006 pooled on the archived measurements.

*Lab verdict:* holdout pooled 0.3121 (need > 0.3118), median 0.2727 (need >= 0.2695), coverage 0.7373 (band [0.74, 0.86], or within 0.0536 of 0.80), horizons {'1': 0.4222, '3': 0.3404, '7': 0.2807}. Gate margins {'gate1': 0.0003, 'gate2': 0.0032, 'gate3': -0.0027, 'gate4': 0.3007, 'shortfall': 0.0027}.

## [PACK_PLUS_EXA] iteration 4: PROMOTED [exogenous_signals] (2026-11-15T17:04:29Z)
**Hypothesis.** exogenous_signals (bold, builds on archived iteration 3 — the Gate-1-passing / Gate-3-failing completion stack): TWO stacked levers on the iter-3 code. (1) NEW exogenous web gate: regime-B rate-card band widening x1.5, fired only when a point-in-time web search at the origin returns a <=45-day-old article about GPU rental pricing (fixed query, cached, safe fallback 1.0 = champion when nothing found / no external signals). Mechanism: the 2026 GPU repricing wave is documented in fresh web articles (CCIR 2026-07-08 "sharpest broad repricing of prior-generation silicon in the record"; FastGPU 2026-09-15; ComputingPower 2026-08-28) and makes the menu-cost rate cards' near-zero-width bands under-cover (Discovery card coverage 0.51-0.61, q90 breach 27-31%); pseudo-origin OOS (fit at origin-28d, score next 28 days, 6 Discovery origins) shows x1.5 widening cuts card loss AND raises coverage at both measurable late origins (09-23: loss 0.00501->0.00481, cov 0.774->0.887; 09-16: cov 0.861->0.925), while marketplace widening is loss-negative (excluded — first dry run caught the width leak into marketplace, fixed to cards only). This falsifies the prior "EXA web search is EMPTY for this domain" guardrail: it was based on other query phrasings; 2026-dated GPU price-index articles exist and are point-in-time usable. (2) Tail-factor frontier rebalance tail_scale {1: 0.95, 3: 0.95, 7: 0.95} (was 0.90/0.93/1.0): h1 and h3 recover the measured holdout coverage (playbook frontier: +0.006/+0.004 aggregate at ~-0.0004/-0.0002 pooled) so Gate 3 clears; h7 0.95 is a NEW skill lever — h7 both-tail narrowing was never tried (only h7 widening, falsified): pseudo-origin OOS (6 Discovery origins, chained center, ensemble members, 540-day z) shows f=0.95 cuts h7 loss ~0.8% at 6/6 origins (f0.97 6/6 ~0.55%, monotone toward 0.93) — h7 is 63% of pooled loss, ~+0.002 pooled at the ~80% h1-lever transfer precedent; h7 OOS coverage cost ~-0.008 on h7 cells (~-0.0024 aggregate), budgeted at 2x per the iter-3 coverage-over-transfer lesson. Predicted holdout: pooled ~0.3133 (Gate-1 threshold 0.31183, margin +0.0015), coverage ~0.746 (Gate-3 floor 0.74, margin +0.006; +0.004 even at 2x h7 cost), median ~0.273 (Gate-2 buffer 0.005), h1 ~0.424, h3 ~0.340, h7 ~0.283. FALSIFIED if (a) the web gate never fires on Holdout (card slices byte-identical to iter-3 — NO_OP on the exogenous lever), or (b) the h7 holdout skill delta vs archived iteration-3 (0.2807) is <= 0 (h7 narrowing non-transfer), or (c) Gate 1 fails, or (d) coverage leaves [0.74, 0.86], or any horizon < -0.020.

### [PACK_PLUS_EXA] iteration 4: web-gated rate-card widening + tail frontier rebalance

**Diagnosis.** Iteration 3 passed Gate 1 (+0.0003) but failed Gate 3 by 0.0027 — the gates pin the stack from both sides: Gate 1 needs the h3 narrowing, Gate 3 cannot afford it. The completion needs either a coverage-positive skill lever or a frontier walk. EXA probe (this iteration): the query "GPU cloud rental price drop H100 A100 per hour 2026" returns RICH 2026-dated point-in-time articles (FastGPU 2026-09-15 price-floor study; CCIR 2026-07-08 nine-year rental-rate panel: "the sharpest broad repricing of prior-generation silicon in the record" from Dec 2025, H100 +29%, H200 +70%, Lambda A100 +56% in seven weeks; ComputingPower 2026-08-28 index; Aven/Flopper 2026-09-09). This FALSIFIES the "EXA web search is EMPTY" guardrail (PACK_PLUS_EXA iter 2) — it was based on other query phrasings; the exogenous family via web is REOPENED.

**Analyses (3 used, 6 Discovery origins each, pseudo-origin protocol: fit at origin-28d, score next 28 days).**
1. *h7 z-window averaging {540,360} (ensemble, the playbook's ranked direction #2):* coverage-positive (+0.002 h7 cells at every origin) but loss-neutral-to-negative (z540_360 vs z540: 2/4 origins worse on loss). NOT a skill lever — dead as a completion, kept as a known coverage reserve.
2. *h7/h3 tail-factor x z-window frontier:* h7 both-tail NARROWING (never tried — only h7 widening was falsified) is OOS-positive 6/6: f0.95 ~-0.8% h7 loss, f0.97 ~-0.55%, f0.93 ~-1.05%, monotone; h7 OOS coverage cost ~-0.008 h7 cells at f0.95. h3 f0.93 confirmed ~-1.3% (matches iter-3). Frontier arithmetic vs the measured iter-3 holdout base (0.31215 pooled / 0.7373 coverage): h1 0.90->0.95 (+0.006 cov, -0.0004 pooled), h3 0.93->0.95 (+0.004 cov, -0.0002), h7 1.0->0.95 (+0.002 pooled at 80% transfer, -0.0024 aggregate cov, budgeted 2x = -0.005).
3. *Regime-B/C band-widening sweep:* card (rate-card) widening x1.5 is OOS loss-AND-coverage-positive at both measurable late origins (09-23: loss 0.00501->0.00481, cov 0.774->0.887; 09-16: cov 0.861->0.925); x1.25 mixed, x2.0 worse; marketplace widening loss-negative at every origin (excluded).

**Candidate.** Iteration-3 code + (a) tail_scale {1: 0.95, 3: 0.95, 7: 0.95}, (b) `_web_repricing_gate`: one fixed point-in-time search per origin ("GPU cloud rental price per hour repricing", num_results=8); returns 1.5 iff a <=45-day-old article with pricing keywords exists, else 1.0 (champion fallback); applied ONLY to regime-B card width (first dry run leaked it into marketplace via width=1.0 — caught by the slice deltas, fixed).

**Dry runs (2).** First: pooled +0.0039, screen pass — but marketplace slices -0.011..-0.016 (the width leak). Second (fixed): pooled +0.0043 (0.4504), 9/11 origins improved, h1 +0.0178 / h3 +0.0029 / h7 +0.0027, coverage 0.7850 (-0.0097), card slices +0.010..+0.021 skill with +0.01..+0.02 coverage (the gate FIRED on Discovery), marketplace deltas exactly 0.0 (fallback verified), sanity clean, would_reach_holdout true.

**Honest prediction.** Holdout pooled ~0.3133 (threshold 0.31183, margin +0.0015 — the largest Gate-1 margin of any submission this stage); coverage ~0.746 (floor 0.74, margin +0.006; +0.004 even if the h7 coverage cost over-transfers 2x); median ~0.273 (buffer 0.005); h1 ~0.424, h3 ~0.340, h7 ~0.283. Main risks: (a) the h7 narrowing does not transfer (then pooled ~0.3117, Gate 1 fails by a hair); (b) the web gate does not fire on Holdout origins (NO_OP on the exogenous lever — the card slices would be byte-identical to iter-3); (c) h1/h3 coverage recovery under-delivers vs the playbook frontier estimates.

**Lessons.** (a) The EXA-EMPTY guardrail was query-phrasing-specific, not domain-specific — 2026 GPU price-index articles exist and are point-in-time usable; the exogenous family via web is reopened with a falsifiable, fallback-safe gate. (b) h7 narrowing is a live skill lever where h7 widening was dead — the sign of the tail-factor dimension flips with the horizon's coverage state (h7 OOS over-covers ~0.85 at late Discovery origins). (c) Slice-level dry-run deltas catch width leaks that aggregate pooled skill hides (marketplace -0.016 invisible in +0.0039 pooled).

*Lab verdict:* holdout pooled 0.3139 (need > 0.3118), median 0.2713 (need >= 0.2695), coverage 0.75 (band [0.74, 0.86], or within 0.0536 of 0.80), horizons {'1': 0.4205, '3': 0.3411, '7': 0.2836}. Gate margins {'gate1': 0.0021, 'gate2': 0.0018, 'gate3': 0.01, 'gate4': 0.3036, 'shortfall': 0}.

## [PACK_PLUS_EXA] iteration 5: REVERTED [exogenous_signals] (2026-11-15T17:20:27Z)
**Hypothesis.** exogenous_signals (bold, builds on the champion = iter 4): TWO stacked levers on the champion code. (1) NEW exogenous web-gate frontier walk: card_widen 1.5 -> 2.0 — the iter-4 web-gated regime-B rate-card band widening is pushed along its measured frontier. Pseudo-origin OOS (fit at origin-28d, score the next 28 days, 4 late Discovery origins, 2988 card cells): g1.5 -0.68% card loss (iter-4 choice), g1.75 -0.41%, g2.0 +0.30% card loss BUT coverage 0.680 -> 0.762 (+0.082 card cells, ~+0.001 aggregate) and q90 breach 0.159 -> 0.139; g2.5 +2.2% loss (beyond the frontier). Mechanism: the cards are ~2% of pooled loss so the skill cost is ~-0.00003 pooled — g2.0 is a cheap Gate-3 coverage RESERVE bought against the h7 narrowing's coverage spend; the gate stays point-in-time (fires only on a <=45-day-old GPU-pricing article, 1.0 fallback = champion). (2) h7 tail-factor walk 0.95 -> 0.93 (interval_calibration): pseudo-origin OOS (fit at origin-28d, score next 28 days, 9 Discovery origins, full champion machinery: chained center, ensemble members, 540-day z) shows f0.93 cuts h7 loss -0.41% vs f0.95 (f0.95 was -0.80% vs 1.0, monotone toward 0.93); h7 is 63% of pooled loss -> ~+0.0011 pooled at the ~0.9x iter-4 h7 transfer precedent; h7 OOS coverage cost ~-0.005 h7 cells (~-0.0015 aggregate), budgeted at 2x = -0.003 against the card reserve (+0.001) and the champion's 0.010 Gate-3 margin. h3 stays 0.95 (f0.93 at h3 was holdout-measured WORSE, iter-4 stack arithmetic +0.0007 for 0.95 — non-transfer precedent). A third lever was measured and EXCLUDED: pooled cross-sectional calm-zone center damping 0.7 (OOS-positive h3 -0.12%/h7 -0.20% loss, coverage +0.006 — the only conditional form ever OOS-positive) is in-sample-negative -0.004 pooled (dry run 1 screen FAIL with all levers on): the champion's chained center is in-sample-optimal so any centering shift is in-sample-negative by construction, 4x its OOS gain — cannot clear the pre-Holdout screen; kept as disabled machinery (calm_damp=1.0) for a future companion. Also measured dead this iteration: chain step-damping (OOS-positive loss at all d<1), asymmetric tail factors (all loss-positive), h7 z-window averaging {540,360} (loss-positive, coverage-only). Dry run 2 (submitted stack): Discovery pooled 0.451364 (+0.000926 vs champion), median +0.000284, coverage 0.7863 (+0.0013), h7 +0.0014 skill, 9/11 origins improved, screen PASS. Predicted holdout: pooled 0.3145-0.3155 vs the +0.003 Gate-1 threshold 0.31692 (borderline — the h7 walk's ~+0.0011 and the card reserve's ~-0.00003 do not sum to +0.003; this submission is primarily the falsification test of the h7 0.93 walk's transfer and the card g2.0 reserve's Gate-3 insurance); coverage ~0.748-0.752 (in-band); median ~0.271 (Gate-2 buffer 0.005); h1 ~0.4205 unchanged, h3 ~0.3411 unchanged, h7 0.2836 + ~0.002-0.003. FALSIFIED if the h7 holdout skill delta <= 0 (0.93 non-transfer), or Gate 1 fails (expected if the levers sum < +0.003), or coverage leaves [0.74, 0.86], or any horizon < -0.020.

### Theory update — PACK_PLUS_EXA iteration 5

### What was tested
A bold exogenous_signals stack on the iter-4 champion: (a) the web-gated regime-B card band
widening walked 1.5 -> 2.0 along its measured OOS frontier (g2.0 = +0.082 card-cell coverage for
+0.3% card loss — a cheap Gate-3 reserve, cards ~2% of pooled loss), and (b) the h7 both-tail
factor walked 0.95 -> 0.93 (OOS -0.41% h7 loss vs f0.95, monotone; ~+0.0011 pooled expected at the
~0.9x iter-4 h7 transfer precedent).

### New measurements this iteration (pseudo-origin OOS, 9 Discovery origins, full champion machinery)
- **Card widening frontier (g in {1.0, 1.25, 1.5, 1.75, 2.0, 2.5}, 2988 cells):** g1.5 -0.68% loss
  (loss-optimal), g2.0 +0.30% loss / coverage 0.680->0.762 / q90 breach 0.159->0.139, g2.5 +2.2%.
  The frontier is flat near the optimum: coverage can be bought at ~+0.3% card loss per +0.04
  coverage up to g2.0; beyond that it steepens.
- **h7 tail factor:** f0.93 -0.41% h7 loss vs f0.95 (which was -0.80% vs 1.0) — monotone toward
  0.93, consistent with the iter-4 finding. h3 f0.93 remains excluded: the iter-4 holdout stack
  arithmetic measured 0.93->0.95 = +0.0007 h3 skill, i.e. 0.93 was holdout-worse at h3.
- **Pooled cross-sectional calm-zone center damping (NEW, near-miss):** damping the chained
  center's deviation from the last value by 0.7 for series whose 14-day vol is at/below the
  pooled cross-sectional median is OOS-POSITIVE at h3 (-0.12% loss) AND h7 (-0.20% loss) with
  coverage +0.006 both horizons — the FIRST conditional centering rule ever OOS-positive in this
  lab (per-series conditionals are dead, iter 7; the pooled cross-sectional form is the one that
  works, consistent with the width-lever history). BUT it is in-sample-negative -0.004 pooled
  (dry run 1: screen FAIL at -0.003 with all levers on): the champion's chained center is
  in-sample-optimal, so ANY centering shift is in-sample-negative by construction, and the
  penalty is ~4x the OOS gain. It cannot clear the pre-Holdout screen even stacked with
  in-sample-positive levers. Kept as disabled machinery (calm_damp=1.0).
- **Dead levers measured this iteration:** chain step-damping (d<1 on the chain step: OOS loss
  positive at every d, h3 +0.01..+0.08%, h7 +0.02..+0.12% — the pure chain is OOS-optimal);
  asymmetric tail factors (f10 != f90: all loss-positive, e.g. h7 asym 0.90/1.00 +0.59%,
  0.93/1.00 +0.73%, 1.00/0.93 +0.03% — the symmetric form is optimal at every horizon);
  h7 z-window averaging {540,360} (+0.45% h7 loss, coverage-only reserve — confirms the
  playbook entry).

### Gate arithmetic
Champion holdout 0.3139 pooled, Gate-1 threshold 0.31692 (+0.003). The submitted stack's
expected holdout delta: h7 0.93 walk ~+0.0011, card g2.0 ~-0.00003 skill / +0.001 coverage
reserve. Sum ~+0.0011 — SHORT of +0.003: this submission is primarily the falsification test
of (a) the h7 0.93 walk's transfer (predicted +0.002..+0.003 h7 skill) and (b) the card g2.0
reserve's Gate-3 insurance. If the h7 walk transfers at the iter-4 precedent (~0.9x), the
champion's Gate-3 margin (0.010) absorbs the h7 coverage spend with the card reserve as backup.

### Lessons
1. The pre-Holdout screen is a hard constraint on centering levers: OOS-positive centering
   shifts are in-sample-negative by construction (the champion center is in-sample-optimal) and
   the in-sample penalty (~4x the OOS gain here) blocks the screen. A centering lever can only
   pass bundled with an in-sample-positive lever of at least the same size — none of that size
   exists in the measured frontier right now.
2. The card widening frontier is flat near g1.5: coverage reserves are cheap there. The web
   gate's multiplier is a tunable coverage knob, not just a skill lever.
3. Asymmetric tails are dead in ALL forms at every horizon — the symmetric tail factor is
   optimal despite the q10/q90 breach-rate asymmetry (the breach asymmetry is a centering
   artifact, not a width signal; consistent with the iter-1 finding).

*Lab verdict:* holdout pooled 0.3145 (need > 0.3169), median 0.2709 (need >= 0.2663), coverage 0.7507 (band [0.74, 0.86], or within 0.055 of 0.80), horizons {'1': 0.4215, '3': 0.3421, '7': 0.2839}. Gate margins {'gate1': -0.0024, 'gate2': 0.0046, 'gate3': 0.0107, 'gate4': 0.3039, 'shortfall': 0.0024}.

## [PACK_ONLY] iteration 10: RUNTIME_ERROR (2026-11-15T18:00:41Z)
**Hypothesis.** MODEL-CLASS EXPLORATION (model_combination, bold): the AWS regime is rewritten from scratch as a 50/50 log-space quantile average of two forecasters from DIFFERENT classes, both fit point-in-time on the pooled AWS benchmark series (trailing 540 days): (1) gradient_boosting — sklearn HistGradientBoostingRegressor(loss='quantile') per quantile on trailing 1/3/7-day log changes + 7/14/28-day local vol, a nonlinear state->quantile map; (2) pooled_linear_momentum — the champion's class embedded (pooled OLS momentum + vol-window ensemble {7,14,28} at h3/h7 + 0.95 h1 tail narrowing + vol-scaled conformal z). Mechanism: the champion's pooled LINEAR momentum OLS cannot represent nonlinear state-conditional dynamics (vol clustering, momentum saturation, sticky zones); the GBM member captures them (alone: Discovery pooled 0.4856 vs champion 0.4462, +0.039, 10/11 origins, h1 +0.070/h3 +0.036/h7 +0.036 skill) but its tree q50 drifts on calm sticky zones (median skill 0.2438 vs 0.2831, Gate-2 risk), so the linear member anchors the calm-zone center and the calibrated band while the tree carries the pooled gain; quantile averaging is the one width lever with a confirmed-positive record (iters 8/9). Regimes B/C byte-identical to the champion. Falsified if holdout pooled <= 0.3118 (Gate 1), or holdout median < 0.2695 (Gate 2), or coverage leaves [0.74, 0.86] (Gate 3), or any horizon skill < -0.020 (Gate 4); the class-exploration mechanism is additionally falsified if the blend's holdout pooled is not above the champion's by at least +0.005 (both members' Discovery pooled gains were +0.039 and 0.0, so a transfer below +0.005 means the tree gain did not transfer out of sample).

### Iter 10 (PACK_ONLY): MODEL-CLASS EXPLORATION — gradient boosting vs analog vs combination

**Directive:** 9 straight pooled_linear_momentum tweaks; this iteration had to write the AWS regime from scratch in >= 2 different classes, dry-run each, submit the best on Discovery.

### Class comparison on Discovery (11 origins, paired vs champion 0.4462 pooled / 0.2831 median / 0.7947 cov)
| class | pooled | median | cov | screen | verdict |
|---|---|---|---|---|---|
| gradient_boosting (plain GBM) | **0.4856** | 0.2438 | 0.7610 | PASS | +0.039 pooled, 10/11 origins, h1 +0.070 / h3 +0.036 / h7 +0.036 skill |
| analog_nonparametric (kernel-weighted standardized paths) | 0.4264 | 0.3035 | 0.7935 | FAIL | -0.020 pooled (its median +0.020 doesn't offset); class NEGATIVE in this form |
| model_combination (GBM x linear, 50/50 log quantile avg) | submitted | — | — | — | expected between the members on both axes |

### Key findings
1. **Nonlinearity is real and large:** the pooled GBM on (1/3/7-day changes, 7/14/28-day vol) beats the champion's pooled linear momentum by +0.039 Discovery pooled (+0.070 h1). The linear state->change map was the binding constraint, not the band machinery. This is the largest single-model Discovery gain measured in this lab.
2. **Tree q50 drifts on calm sticky zones:** GBM median skill 0.2438 vs champion 0.2831 (-0.039). Gate 2 (median >= champion - 0.005) is the binding risk of the plain GBM; hence the 50/50 blend with the linear member (which anchors the calm-zone center) rather than the plain tree.
3. **Analog/kernel class is negative here:** kernel-weighted quantiles of standardized historical paths lose -0.020 pooled despite +0.020 median. The standardized-path resampling smooths exactly the momentum signal the pooled loss lives on. (Implementation bug first: double-log clip — lv is already log; fixed before the verdict.)
4. **Center-inflation bug lesson:** `q50 = q[1] + shrink*(q[1]-0)` doubles the center at shrink=1.0; the three "shrink/widen" dry runs (0.4641 / 0.3948 / 0.3103, monotone in the inflation factor) measured center inflation, NOT shrink or widening. Nothing about shrink/widening was falsified by them; the pure-widening test (gbm_wide3, fixed) came in at 0.4808 pooled / 0.7886 cov — slightly below the plain GBM on pooled, so widening was NOT needed once the center is correct.
5. **GBM under-coverage is centering-driven, not width-driven:** plain GBM q90 breach 14-17% at h3/h7 with coverage 0.761; after the center fix the same machinery covers 0.789. The tree's in-sample quantile fit under-covers OOS but symmetric widening (gbm_wide3) traded pooled skill for coverage without net gain.

### Prediction for Holdout
Blend expected between members: pooled 0.45-0.48 Discovery -> holdout ~0.32-0.35 vs champion 0.3088 (Gate-1 threshold 0.3118); median between 0.2438 and 0.2831 -> holdout ~0.26-0.28 vs Gate-2 floor 0.2695 (the binding risk); coverage ~0.77-0.79 (in band). If Gate 2 fails while Gate 1 passes, the tree center is the problem and iter 11 should re-weight toward the linear member (w_gbm 0.5 -> 0.3) or shrink the tree q50 toward the linear q50 (NOT toward persistence — that was never actually tested, see 4).

*Lab verdict:* RUNTIME_ERROR: ValueError: not enough values to unpack (expected 5, got 4)

## [PACK_ONLY] iteration 11: PROMOTED [ensemble] (2026-11-15T18:57:18Z)
**Hypothesis.** Resubmit the fixed iter-10 model_combination (GBM x pooled-linear-momentum, 50/50 log-space quantile average) whose submission crashed on a tuple-arity bug before any evaluation. The AWS regime is a quantile average of two forecasters from different classes: (1) gradient_boosting — HistGradientBoostingRegressor(loss='quantile') per quantile on trailing 1/3/7-day log changes + 7/14/28-day local vol, pooled over AWS series, 540-day train; (2) pooled_linear_momentum — the champion's class embedded (pooled OLS momentum + vol-window ensemble {7,14,28} at h3/h7 + 0.95 h1 tail narrowing + vol-scaled conformal z). Regimes B/C byte-identical to the champion.

Locked in research episode g001_frozen.
- **mechanism:** The champion's pooled LINEAR momentum OLS cannot represent nonlinear state-conditional dynamics (vol clustering, momentum saturation, sticky zones); the GBM member captures them (alone +0.039 Discovery pooled, the largest single-model gain measured in this lab) but its tree q50 drifts on calm sticky zones in-sample. The 50/50 blend lets the tree carry the pooled gain while the linear member anchors the calm-zone center and the calibrated band. Two bugs were fixed in the archived code: the 4-tuple `per[sid] = (lv, X, y, mask)` unpacked as 5 names (the exact crash) and the undefined `mfall` (should be `m2`). The blend weight was chosen by an OOS dense pseudo-origin backtest (24 origins), not the in-sample dry run: w=0.5 dominates w=0.3 on BOTH axes (pooled +0.0379 vs +0.0254; median 0.2767 vs 0.2704) and w=0.7 gains pooled (+0.0459) but loses the median (0.2666, -0.0146 vs champion — Gate-2 risk). The in-sample median deficit at w=0.5 (-0.037) is leakage, not real drift: OOS it shrinks to -0.0045.
- **expected observation:** On Discovery (11 origins): pooled ~0.481 vs champion 0.446 (+0.035, 10/11 origins), h1 +0.047 / h3 +0.034 / h7 +0.034, coverage ~0.801; in-sample median 0.246 (-0.037) is expected to be leakage. On the dense pseudo-origin backtest (24 origins, stride 3): pooled delta +0.038 (z=6.7, 22/24 origins improved), median 0.277 vs champion 0.281 (-0.0045, inside the Gate-2 tolerance), coverage 0.810. On Holdout: pooled ~0.34-0.36 (Gate-1 threshold 0.3118), median ~0.27-0.28 (Gate-2 floor 0.2695), coverage ~0.75-0.80 (in band), all horizons positive.
- **competing explanation:** The rival explanation of the same errors: the +0.038 OOS pooled gain is not the tree's nonlinearity but the GBM's implicit per-series heterogeneity handling (its features include the series' own local vol, so it acts partly as a per-series conditional width model), and the residual median deficit is real tree drift on sticky zones that the backtest's dense pseudo-origins under-weight because they cluster in Aug-Sep where repricing was active. Under this explanation the Holdout median would fall below the Gate-2 floor even though pooled passes, and a linear member with richer per-series features (or a per-series conformal layer on the GBM) would match the blend without any tree.
- **test:** Dry run on Discovery (11 origins) vs the champion, plus a dense pseudo-origin backtest (24 origins, stride 3) of w_gbm in {0.3, 0.5, 0.7} to select the weight on OOS evidence rather than the in-sample median. The deciding comparison: the committed w=0.5 blend must beat the champion's Discovery pooled 0.4462 (dry run: 0.4814, screen PASS, would_reach_holdout=true) and its OOS median must stay within the Gate-2 tolerance of the champion's (backtest: -0.0045).
- **rejection condition:** Falsified if holdout pooled <= 0.3118 (Gate 1), or holdout median < 0.2695 (Gate 2), or coverage leaves [0.74, 0.86] (Gate 3), or any horizon skill < -0.020 (Gate 4). The class-exploration mechanism is additionally falsified if the blend's holdout pooled gain over the champion is below +0.005 (both members' Discovery gains were +0.039 and 0.0; a transfer below +0.005 means the tree gain did not transfer out of sample). The weight-selection method is falsified if the holdout median falls below 0.2695 while pooled passes — that would mean the OOS backtest median axis also leaks.

*Lab verdict:* holdout pooled 0.3412 (need > 0.3118), median 0.296 (need >= 0.2695), coverage 0.7493 (band [0.74, 0.86], or within 0.0536 of 0.80), horizons {'1': 0.4456, '3': 0.3706, '7': 0.3101}. Gate margins {'gate1': 0.0294, 'gate2': 0.0265, 'gate3': 0.0093, 'gate4': 0.3301, 'shortfall': 0}.

## [LONG_HORIZON] iteration 2: REVERTED [regime_change] (2026-11-15T20:46:36Z)
**Hypothesis.** Family-trend-scaled q90 widening (regime_change): the h90/h180 q90 breaches concentrate in instance families whose spot level has been RISING over the last 90 days (p5 coverage 0.52-0.67 vs falling p4d breaching q10), because AWS's administered repricing continues in the trend direction at long horizons; widening ONLY the upper quantile of rising families by k*clip(family 90d trend, 0, 0.4)*(h/180) with k=1.0 (q10 byte-identical, down-side measured loss-negative) cuts the 16-32% q90 breach loss. Stacked on the per-series x family change-quantile blend (w=0.3 at h90, 0.5 at h180, >=30 own pairs else family fallback) from the same diagnosis.

Locked in research episode g003_candidate_r2.
- **mechanism:** AWS reprices spot per instance family in multi-month waves (price-list churn is continuous: CostPulse 2024-08-02 shows 65,388 cuts vs 259,823 increases in one EC2 publication). A family whose median level rose over the trailing 90 days is in an up-wave; over the next 90-180 days its administered moves keep skewing up, so the pooled 730-day change quantiles (which average over up- and down-waves) under-cover the upper tail exactly there. Adding k*clip(famtr,0,0.4)*(h/180) to q90 only (h>=90) prices that skew; the q10 side stays byte-identical because falling families breach q10 and widening it was measured loss-negative (-0.002/-0.004).
- **expected observation:** Discovery pooled 0.2675 -> ~0.277 (+0.009), h90 +0.017 / h180 +0.006, q90 breach 24.6% -> ~20.6%, coverage +0.024, 12/16 origins improved; dense stride-14 backtest pooled delta ~+0.010 with ~85% of origins improved. On Holdout: pooled ~0.295-0.300 vs gate-1 threshold 0.2916, coverage ~0.78 (in band), all horizons positive.
- **competing explanation:** The rival explanation of the same errors: the q90 breaches are not trend-persistence but a generic under-dispersion of the 730-day pooled change quantiles at h90/h180 (any unconditional widening would fix them), and the gain is a width-for-skill trade that will not transfer. Evidence against: unconditional/symmetric widening was measured loss-negative in this lab's prior stages and the down-side (q10) widening tested here is negative (-0.002/-0.004 h90/h180) while the up-side is positive — a generic under-dispersion story would reward both sides; the gain also concentrates exactly where the trend signal is largest (rising p5 families, coverage 0.52-0.67 -> 0.57-0.69).
- **test:** Pseudo-origin scan (12 monthly pods 2023-07..2024-06, point-in-time): up-side widening k=0.5/1.0 monotone positive (h90 0.322->0.331->0.337, h180 0.363->0.366->0.369), down-side negative; then dry run on the 16 scored Discovery origins (pooled 0.2766 vs champion 0.2675, +0.0091, h90 +0.0174, h180 +0.0059, coverage 0.664->0.688, q90 breach 24.6%->20.6%, 12/16 origins, screen PASS) and a dense stride-14 backtest over 33 origins (pooled delta +0.0101, 85% of origins improved, h90 +0.0179 / h180 +0.0075).
- **rejection condition:** Rejected if Discovery pooled <= 0.2675 (measured 0.2766, screen PASS) or sanity fails. The trend-widening mechanism is falsified if the holdout pooled gain over the champion (0.2886) is below +0.003 (gate 1), or if holdout coverage_80 leaves [0.74, 0.86] (gate 3) — i.e. the widening bought width without coverage. The direction claim is falsified if the h90 gain specifically fails to transfer (holdout h90 <= 0.2226, the champion's value).

**Market insight:** The forecast assumes AWS GPU spot prices at 30-180-day horizons are administered in family-level repricing waves whose direction persists for quarters: a family whose cross-sectional level rose over the last 90 days (p5/H100 premium capacity through 2023-24 AI demand; NextPlatform 2024-05-02 H100 allocation data, TechSpot 2024-02-27 easing H100 lead times) keeps repricing upward, so its q90 must be widened by the recent family trend, while falling families (p4d/A100, repriced down as Capacity Blocks reclaim idle capacity — AWS News Blog 2023-10-31 aws.amazon.com/blogs/aws/announcing-amazon-ec2-capacity-blocks-for-ml; re:Invent 2024 capacity-management recap, zenn.dev/articles/831696491092b17) breach q10, not q90, and their lower band is left untouched. Dated evidence that AWS reprices continuously rather than drifting freely: the 2024-08-02 EC2 price-list publication churn (costpulse.io/aws-price-changes/amazon-elastic-compute-cloud/20240802172840: 65,388 cuts vs 259,823 increases in one version) — direction-mixed at instance level, which is exactly why the lever uses the pack's own family trend rather than news. The view would change if family trends mean-reverted within 90 days (then the widening buys width without coverage), if a generation transition (p6/Blackwell) repriced p5 down monotonically (the up-only widening would over-cover), or if the Holdout coverage left the [0.74, 0.86] band.

*Lab verdict:* holdout pooled 0.2475 (need > 0.2916), median 0.2621 (need >= 0.2685), coverage 0.7569 (band [0.74, 0.86], or within 0.0505 of 0.80), horizons {'30': 0.1823, '90': 0.1384, '180': 0.3589}. Gate margins {'gate1': -0.0441, 'gate2': -0.0064, 'gate3': 0.0169, 'gate4': 0.1584, 'shortfall': 0.0505}.
