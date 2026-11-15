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
point-in-time web search (external_signals.search) — REOPENED as a usable signal source (iter 4).

## Where the loss lives (check before every hypothesis)
- The pooled score is scale-dependent and *fragile*: one exploding cell can annihilate it (PACK_ONLY iter 2:
  a single series with width_rel 1e86 drove pooled skill to -1e84). Read `slices[].share_of_total_loss` and
  `worst_series` first; ~95% of champion loss is AWS hyperscaler_spot (h7 63%, h3 27%, h1 10%).
- Fix the regime that carries the loss. A gain on a 1% slice cannot move the pooled skill — EXCEPT when it
  also buys coverage: the iter-4 card widening (~2% of loss) added +0.01 Gate-3 margin at ~zero pooled cost.
- **Gate 1 needs pooled > champion + 0.003 (now holdout pooled > 0.31692).** Holdout pooled deltas decompose
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
3. Change one thing at a time so the ledger attributes the effect; when stacking CONFIRMED levers, submit
   the stack and attribute via the per-lever deltas already measured on Holdout (PACK_ONLY 6, 8 and
   PACK_PLUS_EXA 1-4 did this cleanly).
4. Record the result, including negative results and their magnitude. Use analyze_discovery to measure
   out-of-sample BEFORE spending a dry run or a submission (budget: ~6 analyses, 4 dry runs per iteration).
5. **Pseudo-origin OOS analysis (fit at origin-28d, score the next 28 days inside Discovery) is the strongest
   pre-screen — but it is NOT a guarantee at h3/h7:** confirmed transfer cases are PACK_ONLY 8, 9 (~1.5-5x),
   PACK_PLUS_EXA 1 chain (~0.7x), PACK_PLUS_EXA 2 w270-at-h1 (~0.7x), PACK_PLUS_EXA 3 h3-f0.93-on-skill
   (~1.9x) and PACK_PLUS_EXA 4 h7-f0.95-on-skill (~0.9x: OOS ~0.8% h7 loss cut -> holdout h7 +0.0029).
   FALSIFIED non-transfer cases: PACK_PLUS_EXA 2 w270-at-h3 (6/6 OOS-positive, holdout h3 -0.0022) and
   PACK_PLUS_EXA 3 h3-f0.93-on-COVERAGE (Discovery predicted -0.0035, holdout cost -0.008). **OOS skill
   transfer does NOT imply OOS coverage transfer: the coverage cost of a narrowing lever over-transfers ~2x.
   Budget Gate 3 at 2x the Discovery-predicted cost.** Short-horizon (h1) z/width signals transfer at ~80%
   prior; h3/h7 recency signals at ~50%; plain both-tail factors at ~90-100%.
6. **In-sample Discovery diagnostics LIE about the median when a lever widens bands** (PACK_ONLY 8:
  in-sample -0.0146, holdout +0.0193) **and about centering** (PACK_PLUS_EXA 1: OLS-q50 deviations are
  in-sample-negative BY CONSTRUCTION while OOS/holdout-positive). A lever that is OOS-positive but
  in-sample-negative CANNOT pass the screen alone — stack it with a confirmed in-sample-positive lever.
7. **Slice-level dry-run deltas catch width leaks that aggregate pooled skill hides** (PACK_PLUS_EXA 4:
  the card widening leaked into marketplace via width=1.0, marketplace slices -0.016 invisible in +0.0039
  pooled; caught and fixed before submission). Always read `slice_deltas_best_first` for exact zeros /
  unexpected signs before submitting.

## HARD guardrails (all measured; cite the falsifying iteration to remove any of them)
- **EXA web search IS usable for this domain (FALSIFIED the old "EXA is EMPTY" guardrail — PACK_PLUS_EXA
  iter 4, PROMOTED):** the query "GPU cloud rental price drop H100 A100 per hour 2026" returns rich
  2026-dated point-in-time articles (FastGPU 2026-09-15, CCIR 2026-07-08, ComputingPower 2026-08-28,
  Aven/Flopper 2026-09-09). The old guardrail (iter 2) was based on three OTHER query phrasings (GPU rental
  price data, AWS spot announcements) that returned stale 2018-2023 articles — it was query-phrasing-specific,
  not domain-specific. Web levers must be: fixed few queries, point-in-time age-gated, fallback-safe
  (multiplier 1.0 = champion when nothing found / no external signals), and pre-screened OOS on the pack.
  hardware_fundamentals/capability_ladder still map to launch prices and benchmark ratios, not to the
  benchmark series' collectors — unusable for level anchoring.
- **Bound every output.** Clip per-series z to a small multiple of pooled z AND clip final log-quantiles to
  a sane band around the last value (log q within +/- 1 of log last). Iteration 2's explosion (h7 pooled
  -1e18) came from unbounded per-series z on a sticky series.
- **Always re-assign per-horizon quantities (z, beta) inside the horizon loop BEFORE any conditional
  override (PACK_PLUS_EXA iter 2 dry run):** dropping the default `z = quantile(...)` line leaked the h3 z
  into h7 (coverage 0.55, screen fail).
- **Never touch the LOWER quantile with a "max".** Iteration 3 falsified "elementwise max of pooled z is
  widening-only": max widens the upper side but RAISES q10, narrowing the lower band. Result: q10 breach
  10.1% -> 10.9%, holdout coverage 0.7408 -> 0.7380, Gate 3 fail. Any upper-tail widening experiment must
  leave z10 byte-identical to the champion.
- **Recent-window residuals do NOT estimate forward tails (falsified three times).** Iterations 2, 3 (recent
  split-conformal z) and 7 (pooled vol-state tail scaling: Discovery +0.0034, Holdout ~-0.0007 marginal).
  The champion's full-window in-sample pooled z is honest in expectation; conditioning its WIDTH on any
  recency/state signal has now failed on Holdout every time it was tried — EXCEPT the plain 270-day trailing
  z window at h1 (PACK_PLUS_EXA 2, holdout h1 +0.0058 CONFIRMED) and unconditional ensemble averaging
  (iterations 8, 9). The distinction: pooled/unconditional window changes transfer at h1; per-series/
  state-conditional ones never do. **State/conditioned width levers are dead; unconditional smoothing
  (ensemble averaging) is the one width lever that WORKS (iterations 8, 9).**
- **The 270-day z window is CONFIRMED at h1 and FALSIFIED at h3 (PACK_PLUS_EXA 2):** holdout h1 +0.0058
  (transfers from 6/6 OOS), holdout h3 -0.0022 (despite 6/6 OOS-positive). At h7 it is OOS-negative 0/6
  (270-day fit over-weights the 1-day lag at long horizons, consistent with the archived beta270 finding).
  Use z_train_days=270 at h1 ONLY. w180 is dominated by w270 (iter-3 analysis, 2/6); w360 dominated 4/6.
- **Chained-residual z at h3/h7 is DEAD (PACK_PLUS_EXA 3 analysis):** the z estimated from the chained
  point forecast's level residuals has z_ratio 1.0 vs the direct z at every vol window and origin
  (deltas -0.03% of loss, 6/6 wash). The chain changes only the center, not the residual dispersion.
  Do not retry chain/direct z blends in any form. (First implementation mixed forecast LEVELS with change
  residuals — garbage z, coverage 0.13; always sanity-check a new z against z_ratio ~1 and coverage ~0.8.)
- **h1 tail-factor refit under the w270 z (f 0.88 vs 0.90): OOS-positive 6/6 but only ~+0.0001 pooled**
  (iter-3 analysis; f0.88 beats f0.90 6/6, b540_z270 beats the champion h1 form ~4.3% loss at every origin).
  Below the noise floor; not worth the coverage cost. The measured iter-9 composition (f0.90) stays.
- **Upper-tail widening via max(full-window, recent split-conformal) z: +0.0004 pooled, -0.0028 coverage.
  Dead lever — do not retry recent-window conformal in any form** (iterations 3, 5).
- **Fixed-factor upper-tail (q90-only) widening is dead in ALL forms** (PACK_ONLY 3, 5 for h7 tails;
  PACK_PLUS_EXA iter 1 analysis for q90-only at factors 1.05-1.20 at every horizon: OOS-negative at every
  origin, e.g. w10 h7 avg -0.0069). The breach rate above 10% does not imply a width lever.
- **Per-series local calibration: catastrophic (iteration 2). Dead lever.**
- **Validation-selected parameter changes that can also select "no change" tend to degenerate to the
  champion on holdout** (iteration 4: shrinkage s=1.0 chosen at every holdout origin -> holdout aggregates
  byte-identical -> Gate 1 fails by construction). If a lever needs a *gain*, it must change something
  unconditionally or its selection rule must be shown (on Discovery) to actually fire.
- **Holdout coverage must stay >= 0.74** (Gate-3 band floor; the 2026-10-24 benchmark sits at 0.7595 under
  the new champion). The old champion was at 0.7514; PACK_ONLY iteration 9's variant and the PACK_PLUS_EXA
  iter-1 stack sat at 0.7500, the iter-2 stack at 0.7451, the iter-3 stack at 0.7373 = FAIL (the stack's
  coverage budget was SPENT until iter 4 rebalanced it to 0.7500). Coverage-neutral-or-widening changes
  only from here; never narrow aggregate coverage below 0.74, and budget any residual narrowing at 2x
  its Discovery-predicted coverage cost.
- **Momentum-beta shrinkage toward zero, validation-selected (iteration 4): DEAD.** Holdout aggregates
  byte-identical to champion (validation always picked s = 1.0); where it fired on Discovery h1 it
  slightly hurt. The pooled OLS beta is already at the out-of-sample optimum.
- **Momentum-beta damping toward zero, unconditional (iteration 7): DEAD.** 0.85/0.70 damping of the h7
  momentum term is wash-to-worse out-of-sample in the pseudo-origin analysis. The pooled OLS beta is at the
  out-of-sample optimum in both selection and unconditional forms.
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
  coverage 0.769 -> 0.808. h7's loss is centering, not width. [The ensemble's window-averaged vol scale
  did fix h7 (+0.0133 holdout); the fixed-factor form remains dead. **h7 tail NARROWING f0.95 is
  CONFIRMED-POSITIVE (PACK_PLUS_EXA 4, holdout h7 +0.0029) — the sign of the tail-factor dimension flips
  with the horizon's coverage state (h7 OOS over-covers ~0.85 at late Discovery origins).**]
- **Ensemble with a drift/persistence member is dead (iteration 5):** 0.8/0.2 momentum+drift quantile
  average: pooled -0.0141, 10/11 origins worse. **Ensemble across perturbed hyperparameters of the SAME
  model is CONFIRMED-POSITIVE (iterations 8, 9) — all members must share the good machinery.**
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
- **aggregator_medians and index_prints are degenerate for this pack (iteration 7 analysis):**
  aggregator_medians is a single source (getdeploying_weekly, reservation-style weekly medians) with no
  overlap to the benchmark series' collectors; index_prints has 0-12 rows. Dead as exogenous signals here.
- **Regime-C q50 centering variants are dead (iteration 7 analysis):** recent-60 median drift and
  zero-pin are a wash on the 4-5 evaluable series.
- **Train-day ensemble members {270,540} are dead (iteration 8 analysis):** the 270-day member is
  mixed-to-negative at h7 (beta270 over-weights the 1-day lag: ~1.9 vs ~1.6 at h7). **avgbeta (averaging
  betas with a single z) is a wash — the ensemble gain lives in averaging the z/vol scale, not the point
  forecast.** [PACK_PLUS_EXA 2 refines this: averaging the z across {540,270} is dominated by the plain
  270-day z at h1; the z-window lever applies at h1 only.]
- **Richer vol-window member sets are dead (iteration 9 analysis):** the 4-member set {7,14,21,28} is
  OOS-NEGATIVE at h3 (avg -0.0026) and h7 (avg -0.0029), negative at every pseudo-origin. {7,14,28} is
  the optimum size; richer is not better. Do not retry member-set expansion. **4-member train-window z
  blends {540,270,90} and {270,90} are likewise dominated by the plain 270-day z at h1 (PACK_PLUS_EXA 2
  analysis); 90-day-only is worse.**
- **h7 z-window averaging {540,360} is coverage-positive (+0.002 h7 cells) but loss-neutral-to-negative
  (PACK_PLUS_EXA 4 analysis): a coverage reserve, not a skill lever.** Known and unused.
- **Marketplace band widening is OOS loss-negative at every origin (PACK_PLUS_EXA 4 analysis) — never
  widen the marketplace regime's bands unconditionally.**
- Pool parameters across series where per-series histories are short (< 60 days).
- Point-in-time only: use rows dated on or before the origin; never hard-code dates or per-origin constants.
- Code: numpy/pandas only, vectorised, < 20 s per origin; guard empty histories and NaNs; always return
  every target series; enforce q10 <= q50 <= q90 and positivity. In analysis code, double-check lag
  indexing against a hand-computed example before trusting results (PACK_PLUS_EXA iter 1: a t+1-vs-t-1
  indexing bug produced skill -1e7 and masqueraded as a falsification until caught; iter 3: a level-vs-
  change residual mix produced coverage 0.13 and masqueraded as a lever).

## Current champion state (PACK_PLUS_EXA iteration 4, PROMOTED)
Discovery pooled 0.4504, median 0.2851, coverage 0.7850. Holdout pooled 0.3139, median 0.2713, coverage
0.7500, horizons 0.4205/0.3411/0.2836. Benchmark 2026-10-24: pooled 0.3339, median 0.2233, coverage 0.7595.
Gate-1 threshold for the next promotion: holdout pooled > 0.31692 (+0.003).
Stack: chained h3/h7 q50 (iter-1) + vol-window ensemble {7,14,28} at every horizon + h1 tail 0.95 + 270-day
z at h1 + tail factors {1: 0.95, 3: 0.95, 7: 0.95} + web-gated regime-B card band widening x1.5 (fires iff
a <=45-day-old GPU-pricing article exists at the origin; 1.0 fallback). Regimes B/C otherwise unchanged.

## CONFIRMED-POSITIVE levers (all MEASURED on Holdout; do not re-measure, stack)
1. **h1 tail narrowing 0.95** (PACK_ONLY 5): holdout h1 +0.0047. In the champion.
2. **Regime-C recent-window change quantiles** (PACK_ONLY 6): holdout ~+0.0005 pooled, coverage +0.0028.
3. **Vol-window ensemble {7,14,28} at h3/h7** (PACK_ONLY 8, PROMOTED): holdout h3 +0.0124, h7 +0.0133,
   pooled +0.0122, median +0.0193, coverage +0.0106.
4. **h1 vol-window ensemble + tail re-center** (PACK_ONLY 9 / PACK_PLUS_EXA 1): holdout h1 +0.0134.
5. **Chained (iterated 1-day) q50 at h3/h7** (PACK_PLUS_EXA 1): holdout h3 +0.0010, h7 +0.0006.
6. **270-day trailing z window at h1** (PACK_PLUS_EXA 2): holdout h1 +0.0058. At h1 ONLY.
7. **h3 both-tail narrowing f0.93** (PACK_PLUS_EXA 3): holdout h3 +0.0025 skill (transferred ~1.9x OOS),
   coverage -0.008 (~2x its Discovery-predicted -0.0035).
8. **Web-gated regime-B card band widening x1.5** (PACK_PLUS_EXA 4): card slices +0.010..+0.021 skill,
   +0.01..+0.02 coverage on Discovery; holdout coverage landed 0.7500 in-band with the whole stack.
9. **h7 both-tail narrowing f0.95** (PACK_PLUS_EXA 4): holdout h7 +0.0029 (OOS ~0.8% h7 loss cut, 6/6
   origins, transferred ~0.9x). h7 widening remains dead — the sign flips with the coverage state.
10. **Stack arithmetic (measured on Holdout):** the iter-4 stack = iter-3 base (0.31215/0.7373) +
    h1 0.90->0.95 (-0.0017 h1 skill, +0.006 cov) + h3 0.93->0.95 (+0.0007 h3, +0.004 cov) + h7 1.0->0.95
    (+0.0029 h7, ~-0.005 cov) + card widening = 0.3139 pooled / 0.7500 coverage. All four gates pass.

## Ranked untried directions (the next promotion needs +0.003 pooled on top of 0.3139)
1. **Web-signal refinement (exogenous_signals):** the gate is currently binary (fresh article -> x1.5).
   Untried: (a) age-weight the multiplier (1.5 -> 1.2 for 45-90-day-old articles); (b) use article CONTENT
   (price-direction keywords) to sign a small q50 drift on the cards; (c) a *conditional* marketplace form
   (its unconditional widening is loss-negative). All must be pre-screened OOS and stay fallback-safe.
2. **h7 tail-factor walk toward 0.93** (interval_calibration): OOS monotone toward 0.93 (~-1.05% h7 loss
   vs -0.8% at 0.95) — ~+0.001 pooled more, but the coverage cost doubles (~-0.005 more); the champion's
   Gate-3 slack is ~0.01. Measure the exact OOS frontier first; budget at 2x.
3. **h3 f0.93 restore + h7 f0.93 combo** (interval_calibration): recovers the iter-3 h3 gain (+0.0025)
   but re-spends ~0.008 coverage — only viable bundled with a coverage-positive lever of the same size
   (e.g. the h7 z-window averaging {540,360} reserve, +0.002 h7 cells).
4. **Log-band clip (last log value +/- 1.0) bundled in (robustness):** pure safety, expected NO_OP on
   scored cells; bundle with any stack, never submit alone.
5. **Regime-C recent/full quantile blend at h7 (regime_change):** OOS 5/6 positive but ~2% of loss
   (~+0.00007 pooled) — below the noise floor, only worth bundling if free.
6. **cross_series pooled treatment of the sticky p5/p5e zones:** per-series conditionals are dead (iter 7);
   only a pooled/unconditional treatment could work — low prior.

## Notebook (append-only ledger of what was tried — key entries; full history rendered by the lab)
- PACK_ONLY iters 0-9: see the ledger. Champion was iter 8 (vol-window ensemble at h3/h7).
- **PACK_PLUS_EXA iters 1-3 (REVERTED):** chain (h3/h7 q50, holdout +0.0020 pooled), w270-z@h1
  (+0.0058 h1 CONFIRMED; at h3 FALSIFIED -0.0022), h3 f0.93 (+0.0025 h3 skill, coverage -0.008).
  Iter 3 passed Gate 1 (+0.0003) but failed Gate 3 (-0.0027) — the gates pinned the stack from both sides.
- **PACK_PLUS_EXA iter 4 (PROMOTED, exogenous_signals bold, builds on 3):** web-gated regime-B card
  widening x1.5 (EXA-EMPTY guardrail falsified — 2026 GPU price-index articles exist and are point-in-time
  usable) + tail frontier rebalance {1: 0.95, 3: 0.95, 7: 0.95} (h7 narrowing NEW skill lever, OOS 6/6,
  holdout h7 +0.0029; h1/h3 coverage recovery per the measured frontier). Holdout 0.3139 pooled
  (Gate 1 +0.0021), 0.2713 median (Gate 2 +0.0018), 0.7500 coverage (Gate 3 +0.01), all horizons positive
  (Gate 4). First promotion of this stage. Dry-run lesson: slice deltas caught the card widening leaking
  into marketplace (width=1.0 default) — fixed to cards-only before submission.
