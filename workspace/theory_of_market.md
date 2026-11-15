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
