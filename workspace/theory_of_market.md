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
