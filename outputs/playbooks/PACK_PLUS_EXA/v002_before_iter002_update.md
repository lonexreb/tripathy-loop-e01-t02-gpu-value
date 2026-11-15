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
point-in-time web search (external_signals.search) — an untried exogenous capability.

## Where the loss lives (check before every hypothesis)
- The pooled score is scale-dependent and *fragile*: one exploding cell can annihilate it (PACK_ONLY iter 2:
  a single series with width_rel 1e86 drove pooled skill to -1e84). Read `slices[].share_of_total_loss` and
  `worst_series` first; ~95% of champion loss is AWS hyperscaler_spot (h7 63%, h3 27%, h1 10%).
- Fix the regime that carries the loss. A gain on a 1% slice cannot move the pooled skill.
- **Gate 1 needs pooled > champion + 0.003.** A lever confined to h1 (~10% of loss share) cannot clear it
  alone even when it transfers perfectly (PACK_ONLY iter 9: h1 holdout +0.0134, pooled only +0.0013).
  Single-horizon levers must be stacked with a second confirmed lever, or target h7 (63% of loss) directly.
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
   already measured on Holdout — PACK_ONLY iterations 6 and 8 and PACK_PLUS_EXA 1 did this cleanly.)
4. Record the result, including negative results and their magnitude. Use analyze_discovery to measure
   out-of-sample BEFORE spending a dry run or a submission (budget: ~6 analyses, 4 dry runs per iteration).
   NOTE: the analysis budget exhausts fast; plan analyses to answer several questions per call.
5. **Pseudo-origin OOS analysis (fit at origin-28d, score the next 28 days inside discovery) is the strongest
   pre-screen available.** Confirmed positive cases: PACK_ONLY iters 8, 9 (~1.5-5x transfer) and PACK_PLUS_EXA
   iter 1 chain (~0.7x transfer, still positive). Negative case: PACK_ONLY iter 7 (calm-origin WIDTH
   conditioning gains did not transfer). Prefer unconditional smoothing over conditional rescaling.
6. **In-sample Discovery diagnostics LIE about the median when a lever widens bands**: PACK_ONLY iter 8's
   Discovery in-sample median dropped -0.0146 while the holdout median ROSE +0.0193. **And they lie about
   centering too (PACK_PLUS_EXA iter 1):** a q50 that deviates from the pooled-OLS q50 is in-sample-negative
   BY CONSTRUCTION (OLS minimises in-sample loss) while being OOS/holdout-POSITIVE (chain: in-sample -0.0009,
   holdout +0.0010/+0.0006). A lever that is OOS-positive but in-sample-negative CANNOT pass the screen
   alone — stack it with a confirmed in-sample-positive lever to clear the screen.

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
  are dead; unconditional smoothing (ensemble averaging) is the one width lever that WORKS (iterations 8, 9).**
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
- **Holdout coverage must stay >= 0.74** (Gate-3 band floor; the 2026-10-24 benchmark sits at 0.7526 under
  the champion). The champion is at 0.7514; PACK_ONLY iteration 9's variant and the PACK_PLUS_EXA iter-1
  stack both sit at 0.7500 (in-band). Coverage-neutral-or-widening changes only; never narrow aggregate
  coverage below 0.74.
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
  coverage 0.769 -> 0.808. h7's loss is centering, not width. [The ensemble's window-averaged vol scale
  is NOT a fixed-factor widening and did fix h7 (+0.0133 holdout); the fixed-factor form remains dead.]
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
  forecast.**
- **Richer vol-window member sets are dead (iteration 9 analysis):** the 4-member set {7,14,21,28} is
  OOS-NEGATIVE at h3 (avg -0.0026) and h7 (avg -0.0029), negative at every pseudo-origin. {7,14,28} is
  the optimum size; richer is not better. Do not retry member-set expansion.
- Pool parameters across series where per-series histories are short (< 60 days).
- Point-in-time only: use rows dated on or before the origin; never hard-code dates or per-origin constants.
- Code: numpy/pandas only, vectorised, < 20 s per origin; guard empty histories and NaNs; always return
  every target series; enforce q10 <= q50 <= q90 and positivity. In analysis code, double-check lag
  indexing against a hand-computed example before trusting results (PACK_PLUS_EXA iter 1: a t+1-vs-t-1
  indexing bug produced skill -1e7 and masqueraded as a falsification until caught).

## Current champion state (PACK_ONLY iteration 8, PROMOTED; builds on iterations 1, 5, 6)
Discovery pooled 0.4462, median 0.2831, coverage 0.7947, breaches 9.0%/11.5%. Holdout pooled 0.3088,
median 0.2745, coverage 0.7514, horizons 0.4030/0.3369/0.2801. Gate margins all positive
(gate1 +0.0092, gate2 +0.0243, gate3 +0.0156, gate4 +0.3001). Benchmark 2026-10-24: pooled 0.3252,
median 0.2086, coverage 0.7526.
Regimes: A AWS spot = pooled momentum OLS (train 540) + local-vol conformal with **quantiles averaged
across vol_wins {7,14,28} at h3/h7** (h1 single-window + 0.95 tail narrowing); B rate cards = empirical
change quantiles + stickiness pin + IQR width; C marketplace/non-card = trailing-60-day change quantiles
(full-history fallback), kappa = 0.

## CONFIRMED-POSITIVE levers (all MEASURED on Holdout; do not re-measure, stack)
1. **h1 tail narrowing 0.95** (PACK_ONLY 5): holdout h1 +0.0047. In the champion.
2. **Regime-C recent-window change quantiles** (PACK_ONLY 6): holdout ~+0.0005 pooled, coverage +0.0028.
3. **Vol-window ensemble {7,14,28} at h3/h7** (PACK_ONLY 8, PROMOTED): holdout h3 +0.0124, h7 +0.0133,
   pooled +0.0122, median +0.0193, coverage +0.0106. Transferred at ~5x its Discovery magnitude.
4. **h1 vol-window ensemble + 0.90 re-center** (PACK_ONLY 9; re-measured identically in PACK_PLUS_EXA 1):
   holdout h1 +0.0134, pooled +0.0013, coverage 0.7500 in-band. NOT in the champion (sub-threshold alone).
5. **Chained (iterated 1-day) q50 at h3/h7** (PACK_PLUS_EXA 1): g[0]=last log value,
   g[j+1]=g[j]+beta1.(g[j]-g[j-1], g[j]-g[j-3], g[j]-g[j-7]) with actual history for negative j, inside the
   ensemble members. Holdout h3 +0.0010, h7 +0.0006 (chain part of the stack). In-sample Discovery NEGATIVE
   (-0.0009) — the direct pooled-OLS q50 is in-sample-optimal by construction; the chain is OOS-positive
   (5/5 origins, h3 +0.0015 / h7 +0.0011 avg, coverage-neutral) and holdout-positive. NOT in the champion.
6. **The full stack (4 + 5) measured on Holdout (PACK_PLUS_EXA iter 1, REVERTED on Gate 1 only, shortfall
   0.001):** pooled 0.3108 (+0.0020), median 0.2717 (-0.0029, Gate 2 pass), coverage 0.7500 (in-band),
   h1 0.4164 / h3 0.3379 / h7 0.2807. Gates 2/3/4 pass with margin. **The stack is +0.001 short of Gate 1.**
   Stacking confirmed levers adds up cleanly (iterations 6, 8, PACK_PLUS_EXA 1).

## Ranked untried directions (the lab decides stopping; these are the next hypotheses)
The stack of all confirmed levers is +0.0020 vs the +0.003 Gate-1 threshold: **any new confirmed lever worth
>= +0.001 pooled completes the stack.** Ranked by expected value:
1. **exogenous_signals (EXA web search — the stage's untried capability):** point-in-time search for AWS
   spot price change announcements / capacity crunches to shift the regime-A q50 or widen bands on flagged
   series. Must be pooled/unconditional in application (per-series conditional rules are dead),
   point-in-time cached, and shown to fire on Discovery before submission. If positive, stack with levers 4+5.
2. **exogenous_signals (pack tables): hardware_fundamentals / capability_ladder level anchoring for the
   marketplace and non-AWS hyperscaler q50** (bias_rel -0.05/-0.06 at h3/h7). Small slice (~4% of loss) but
   the bias is real and persistent; check the ladder maps to the benchmark gpu_models before spending anything.
   If positive, stack with levers 4+5.
3. **horizon_structure: chained-residual z** — the chain's band is still the direct-h residual quantile; a
   z estimated from the chained point forecast's residuals is the natural completion of the confirmed chain
   lever. Measure OOS first; blends of chain/direct z were not tested.
4. **robustness: log-band clip (last log value +/- 1.0) bundled into the next candidate.** Pure safety,
   expected NO_OP on scored cells; bundle, never submit alone.
5. **cross_series: pooled cross-AZ residual correlation for the worst sticky series** (p5/p5e use1-az1/az4
   coverage 0.97, skill -0.7): their loss is WIDTH on calm sticky zones; any per-series conditional is dead
   (iteration 7), so only a pooled/unconditional treatment could work — low prior, measure OOS first.
6. **ensemble: weighted member blend (e.g. 0.25/0.5/0.25 centred on 14) — LOW prior:** iteration 8 measured
   the half-weighted blend ~equal to the plain average; the plain average is simpler and confirmed.

## Notebook (append-only ledger of what was tried — key entries; full history rendered by the lab)
- PACK_ONLY iters 0-9: see the ledger. Champion = iter 8 (vol-window ensemble at h3/h7). Iter 9 (h1 ensemble +
  0.90) reverted on Gate 1 only; its lever is confirmed and archived.
- **PACK_PLUS_EXA iter 1 (REVERTED, Gate 1 shortfall 0.001, horizon_structure bold, builds on champion):**
  chained h3/h7 q50 (OOS 5/5 origins, h3 +0.0015 / h7 +0.0011 avg, coverage-neutral) stacked with the iter-9
  h1 ensemble lever to clear the in-sample-negative screen (stack Discovery +0.000058). Holdout: pooled
  +0.0020, h1 +0.0134 (lever 4 exact repeat), h3 +0.0010, h7 +0.0006 (chain transferred at ~0.7x OOS),
  coverage 0.7500 in-band, median -0.0029 (Gate 2 pass). LESSONS: (a) the chain lever is CONFIRMED and
  archived — stack, do not re-measure; (b) the in-sample/OOS inversion extends from width to centering:
  OLS-q50 deviations are in-sample-negative by construction, so a centering lever needs a confirmed
  in-sample-positive partner to pass the screen; (c) the pre-screen transferred again (4th consecutive
  positive case); (d) q90-only widening falsified OOS at every factor/horizon; (e) check lag indexing in
  analysis code by hand before trusting magnitudes like -1e7.
