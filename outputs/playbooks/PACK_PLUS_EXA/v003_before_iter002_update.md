# Scientist Playbook — GPU cloud rental price forecaster (stage PACK_PLUS_EXA)

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
with forecasts that would also be honest at the frozen 2026-11-06 origin. Stage PACK_PLUS_EXA adds
point-in-time web search (external_signals.search) — measured EMPTY for this domain (see guardrails).

## Where the loss lives (check before every hypothesis)
- The pooled score is scale-dependent and *fragile*: one exploding cell can annihilate it (PACK_ONLY iter 2:
  a single series with width_rel 1e86 drove pooled skill to -1e84). Read `slices[].share_of_total_loss` and
  `worst_series` first; ~95% of champion loss is AWS hyperscaler_spot (h7 63%, h3 27%, h1 10%).
- Fix the regime that carries the loss. A gain on a 1% slice cannot move the pooled skill.
- **Gate 1 needs pooled > champion + 0.003 (holdout pooled > 0.31183).** Holdout pooled deltas decompose
  roughly as horizon-skill-delta x loss share: +0.006 h1 skill ~ +0.0006 pooled; +0.002 h3 ~ +0.0005;
  +0.001 h7 ~ +0.0006. Stack levers until the SUM clears +0.003 with margin.
- **Read `mean_q50_error`/`bias_rel` vs `mean_width_rel` before choosing a lever.** h7 under-coverage was a
  CENTERING problem (bias_rel -0.012 at h7), not a width problem; the calm-zone worst series lose on WIDTH
  (coverage 0.97), not q50 bias.
- Breach-rate asymmetry (q90 11.5% vs q10 9.0%) is NOT a width signal: q90-only widening is OOS-negative
  (PACK_PLUS_EXA iter 1 analysis). The champion's q90 is OOS-optimal despite the breach rate.

## Method
1. Diagnose from Discovery only: per-slice skill, breach asymmetry, worst series (bias vs width).
2. State ONE falsifiable hypothesis with a mechanism and a predicted sign.
3. Change one thing at a time so the ledger attributes the effect; keep everything else identical.
   (Exception: when stacking CONFIRMED levers, submit the stack and attribute via the per-lever deltas
   already measured on Holdout — PACK_ONLY 6, 8 and PACK_PLUS_EXA 1, 2 did this cleanly.)
4. Record the result, including negative results and their magnitude. Use analyze_discovery to measure
   out-of-sample BEFORE spending a dry run or a submission (budget: ~6 analyses, 4 dry runs per iteration).
5. **Pseudo-origin OOS analysis (fit at origin-28d, score the next 28 days inside Discovery) is the strongest
   pre-screen — but it is NOT a guarantee at h3/h7:** confirmed transfer cases are PACK_ONLY 8, 9 (~1.5-5x),
   PACK_PLUS_EXA 1 chain (~0.7x) and PACK_PLUS_EXA 2 w270-at-h1 (~0.7x). FALSIFIED non-transfer case:
   PACK_PLUS_EXA 2 w270-at-h3 (6/6 OOS-positive, holdout h3 -0.0022). Short-horizon (h1) z/width signals
   transfer; h3/h7 recency signals do not. Treat h3/h7 OOS wins as ~50% prior, h1 wins as ~80%.
6. **In-sample Discovery diagnostics LIE about the median when a lever widens bands** (PACK_ONLY 8:
  in-sample -0.0146, holdout +0.0193) **and about centering** (PACK_PLUS_EXA 1: OLS-q50 deviations are
  in-sample-negative BY CONSTRUCTION while OOS/holdout-positive). A lever that is OOS-positive but
  in-sample-negative CANNOT pass the screen alone — stack it with a confirmed in-sample-positive lever.

## HARD guardrails (all measured; cite the falsifying iteration to remove any of them)
- **EXA web search is EMPTY for this domain (PACK_PLUS_EXA iter 2):** three point-in-time query phrasings
  (GPU rental price data, AWS spot announcements) return zero or stale 2018-2023 articles. The
  exogenous_signals-via-web family is closed; hardware_fundamentals/capability_ladder map to launch prices
  and benchmark ratios, not to the benchmark series' collectors — also unusable for level anchoring.
- **Bound every output.** Clip per-series z to a small multiple of pooled z AND clip final log-quantiles to
  a sane band around the last value (log q within +/- 1 of log last). Iteration 2's explosion (h7 pooled
  -1e18) came from unbounded per-series z on a sticky series.
- **Always re-assign per-horizon quantities (z, beta) inside the horizon loop BEFORE any conditional
  override (PACK_PLUS_EXA iter 2 dry run):** dropping the default `z = quantile(...)` line leaked the h3 z
  into h7 (coverage 0.55, screen fail). The sanity screen catches it, but only after a dry run is spent.
- **Never touch the LOWER quantile with a "max".** Iteration 3 falsified "elementwise max of pooled z is
  widening-only": max widens the upper side but RAISES q10, narrowing the lower band. Any upper-tail
  widening experiment must leave z10 byte-identical to the champion.
- **Recent-window residuals do NOT estimate forward tails (falsified three times).** Iterations 2, 3 (recent
  split-conformal z) and 7 (pooled vol-state tail scaling). Conditioning the champion's WIDTH on any
  recency/state signal has failed on Holdout every time — EXCEPT the plain 270-day trailing z window at h1
  (PACK_PLUS_EXA 2, holdout h1 +0.0058 CONFIRMED) and unconditional ensemble averaging (iterations 8, 9).
  The distinction: pooled/unconditional window changes transfer at h1; per-series/state-conditional ones never do.
- **The 270-day z window is CONFIRMED at h1 and FALSIFIED at h3 (PACK_PLUS_EXA 2):** holdout h1 +0.0058
  (transfers from 6/6 OOS), holdout h3 -0.0022 (despite 6/6 OOS-positive). At h7 it is OOS-negative 0/6
  (270-day fit over-weights the 1-day lag at long horizons, consistent with the archived beta270 finding).
  Use z_train_days=270 at h1 ONLY.
- **Upper-tail widening via max(full-window, recent split-conformal) z: dead (iterations 3, 5).**
- **Fixed-factor upper-tail (q90-only) widening is dead in ALL forms** (PACK_ONLY 3, 5; PACK_PLUS_EXA 1).
- **Per-series local calibration: catastrophic (iteration 2). Dead lever.**
- **Validation-selected parameter changes that can also select "no change" degenerate to the champion on
  holdout** (iteration 4). If a lever needs a gain, it must change something unconditionally.
- **Holdout coverage must stay >= 0.74** (Gate-3 floor). Champion 0.7514; the iter-1/iter-2 stacks sit at
  0.7500/0.7451 (in-band but with shrinking slack — the w270 h1 narrowing costs ~0.006 coverage).
  Coverage-neutral-or-widening changes only from here.
- **Momentum-beta shrinkage/damping toward zero: DEAD (iterations 4, 7).** The pooled OLS beta is at the
  out-of-sample optimum in both selection and unconditional forms. avgbeta is a wash (iter 8).
- **In-sample mean residuals do NOT persist forward (iterations 5, 7).** Do not "fix" bias with in-sample drift.
- **Per-series conditional width rules are dead (iteration 7).** Only pooled/unconditional rules were ever competitive.
- Cross-collector pull (kappa > 0) and cross-AZ consensus: falsified (iteration 1). Dead.
- **Per-instance-type (cross_series) hierarchy is dead (iteration 5).**
- **Ensemble with a drift/persistence member is dead (iteration 5).** Ensemble across perturbed
  hyperparameters of the SAME model is CONFIRMED (iterations 8, 9) — all members share the good machinery.
- **Calm-regime z narrowing is dead (iteration 5).** **Long-lag momentum dead (iteration 6).**
- **Regime-B rate-card spread floor dead (iteration 6).** **aggregator_medians/index_prints degenerate (iter 7).**
- **Regime-C q50 centering variants dead (iteration 7).** **Richer vol-window member sets dead (iteration 9):
  {7,14,28} is the optimum.** **4-member train-window blends {540,270,90} and {270,90} are dominated by the
  plain 270-day z at h1 (PACK_PLUS_EXA 2 analysis); 90-day-only is worse.**
- Pool parameters across series where per-series histories are short (< 60 days).
- Point-in-time only: use rows dated on or before the origin; never hard-code dates or per-origin constants.
- Code: numpy/pandas only, vectorised, < 20 s per origin; guard empty histories and NaNs; always return
  every target series; enforce q10 <= q50 <= q90 and positivity. Double-check lag indexing against a
  hand-computed example before trusting analysis magnitudes (PACK_PLUS_EXA 1: a t+1-vs-t-1 bug produced
  skill -1e7 and masqueraded as a falsification until caught).

## Current champion state (PACK_ONLY iteration 8, PROMOTED)
Discovery pooled 0.4462, median 0.2831, coverage 0.7947. Holdout pooled 0.3088, median 0.2745, coverage
0.7514, horizons 0.4030/0.3369/0.2801. Gate-1 threshold for promotion: holdout pooled > 0.31183.
Regimes: A AWS spot = pooled momentum OLS (train 540) + local-vol conformal with quantiles averaged across
vol_wins {7,14,28} at h3/h7 (h1 single-window + 0.95 tail narrowing); B rate cards = empirical change
quantiles + stickiness pin + IQR width; C marketplace/non-card = trailing-60-day change quantiles, kappa = 0.

## CONFIRMED-POSITIVE levers (all MEASURED on Holdout; do not re-measure, stack)
1. **h1 tail narrowing 0.95** (PACK_ONLY 5): holdout h1 +0.0047. In the champion.
2. **Regime-C recent-window change quantiles** (PACK_ONLY 6): holdout ~+0.0005 pooled, coverage +0.0028.
3. **Vol-window ensemble {7,14,28} at h3/h7** (PACK_ONLY 8, PROMOTED): holdout h3 +0.0124, h7 +0.0133,
   pooled +0.0122, median +0.0193, coverage +0.0106.
4. **h1 vol-window ensemble + 0.90 re-center** (PACK_ONLY 9; re-measured in PACK_PLUS_EXA 1): holdout h1
   +0.0134, pooled +0.0013, coverage 0.7500 in-band. NOT in the champion.
5. **Chained (iterated 1-day) q50 at h3/h7** (PACK_PLUS_EXA 1): holdout h3 +0.0010, h7 +0.0006. In-sample
   Discovery NEGATIVE (-0.0009) — needs an in-sample-positive partner to pass the screen. NOT in the champion.
6. **270-day trailing z window at h1** (PACK_PLUS_EXA 2): holdout h1 +0.0058 (attribution: iter-2 h1 0.4222
   vs iter-1 0.4164, both stacks share everything else). OOS 6/6 origins, ~1.8% h1 loss cut. At h3 the same
   lever is FALSIFIED (holdout -0.0022) — apply at h1 ONLY. NOT in the champion.
7. **Stack arithmetic (measured, PACK_PLUS_EXA 1+2):** iter-1 stack (4+5) = holdout pooled +0.0020
   (0.31084); + w270-h1-only = expected ~+0.0026 (0.3114) — still ~0.0004 short of Gate 1. The final
   iteration needs one more lever worth >= +0.0004 pooled ON TOP of the h1-only refinement.

## Ranked untried directions (iteration 3 of 3 is the last planned; two reverts in a row -> BOLD)
The completion stack is iter-1 (chain + h1 ensemble/0.90) + w270-z at h1 ONLY ~= +0.0026 pooled. Ranked
completions to reach +0.003 (measure OOS first, then stack; submit the full stack as one bold candidate):
1. **h1 tail-factor refit under the w270 z (interval_calibration x ensemble interaction):** the 0.90 factor
   was tuned for the 540-day z ensemble (iter 9); the w270 z is ~7% narrower, so the OOS optimum of
   f x z-window may sit at f closer to 0.95-1.0. Sweep f in {0.88, 0.90, 0.95, 1.0} x z in {540, 270} at h1
   OOS. h1 signals transfer at ~80% prior — highest-EV untried lever.
2. **Chained-residual z at h3/h7 (horizon_structure):** the chain's band is still the direct-h residual
   quantile; a z estimated from the chained point forecast's residuals is the natural completion of lever 5.
   Blends of chain/direct z were never tested. h3/h7 OOS wins transfer at ~50% prior — measure, do not assume.
3. **Log-band clip (last log value +/- 1.0) bundled in (robustness):** pure safety, expected NO_OP on scored
   cells; bundle with the stack, never submit alone.
4. **Regime-C recent/full quantile blend at h7 (regime_change):** OOS 5/6 positive but the slice is ~2% of
   loss (~+0.00007 pooled) — below the noise floor, only worth bundling if free.
5. **cross_series pooled treatment of the sticky p5/p5e zones:** per-series conditionals are dead (iter 7);
   only a pooled/unconditional treatment could work — low prior.

## Notebook (append-only ledger of what was tried — key entries; full history rendered by the lab)
- PACK_ONLY iters 0-9: see the ledger. Champion = iter 8 (vol-window ensemble at h3/h7).
- **PACK_PLUS_EXA iter 1 (REVERTED, Gate 1 shortfall 0.001, horizon_structure bold):** chained h3/h7 q50
  stacked with the iter-9 h1 ensemble lever. Holdout pooled +0.0020, h1 +0.0134, h3 +0.0010, h7 +0.0006,
  coverage 0.7500, median -0.0029 (Gate 2 pass). Chain CONFIRMED and archived.
- **PACK_PLUS_EXA iter 2 (REVERTED, Gate 1 shortfall 0.001, interval_calibration incremental, builds on 1):**
  270-day trailing z at h1/h3 stacked on the iter-1 chain. EXA search probed first: EMPTY (exogenous family
  via web closed). OOS pre-screen: w270 dominates {540,270} averaging, 90-day and blends at h1/h3; h7 0/6.
  Dry run +0.0016 pooled (h1 +0.0199). Holdout: pooled +0.0019 (0.31075), h1 +0.0191 (w270 h1 part
  +0.0058 CONFIRMED), h3 -0.0012 (w270 h3 part -0.0022 FALSIFIED — 6/6 OOS-positive did not transfer),
  h7 +0.0006 (chain), coverage 0.7451 in-band, median +0.0007. One dry run was spent on a z-leak bug
  (default z assignment dropped -> h3 z leaked into h7, coverage 0.55); caught by the screen, fixed.
  LESSONS: (a) h1 z-window recency transfers, h3 does not — apply window levers at h1 only unless the
  horizon is directly confirmed; (b) the stack arithmetic is now exact: iter-1 + w270-h1 ~= +0.0026,
  Gate 1 needs one more ~+0.0004 lever; (c) averaging is NOT the universal ensemble form — for the vol
  scale it was, for the z window a plain shorter window dominates; sweep the form first.
