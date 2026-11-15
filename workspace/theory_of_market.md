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
