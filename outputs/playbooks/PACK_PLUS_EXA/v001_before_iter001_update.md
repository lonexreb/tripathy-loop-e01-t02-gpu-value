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
- ~95% of champion loss is AWS hyperscaler_spot (h7 63%, h3 27%, h1 10%). Fix that regime first.
- **Gate 1 needs pooled > champion + 0.003.** Loss-share arithmetic: a lever confined to h1 (~10%) transfers
  at most ~+0.0013 pooled; h3+h7 (~90%) is where Gate 1 can be cleared. Single-horizon levers must be stacked.
- **Read `mean_q50_error`/`bias_rel` vs `mean_width_rel` before choosing a lever.** h7 loss is centering
  (bias_rel -0.012), not width; calm-zone worst series lose on WIDTH (coverage 0.97), not q50 bias.
- Breach-rate asymmetry (q90 11.5% vs q10 9.0%) is NOT a width signal: q90-only widening is OOS-negative
  (PACK_PLUS_EXA iter 1 analysis). The champion's q90 is OOS-optimal despite the breach rate.

## Method
1. Diagnose from Discovery only: per-slice skill, breach asymmetry, worst series (bias vs width).
2. State ONE falsifiable hypothesis with a mechanism and a predicted sign.
3. Change one thing at a time; when stacking CONFIRMED levers, submit the stack and attribute via the
   per-lever Holdout deltas already measured (iterations 6, 8, PACK_PLUS_EXA 1 did this cleanly).
4. Record the result, including negative results and their magnitude. Use analyze_discovery to measure
   out-of-sample BEFORE spending a dry run or a submission (budget: ~6 analyses, 4 dry runs per iteration).
   Plan analyses to answer several questions per call.
5. **Pseudo-origin OOS analysis (fit at origin-28d, score the next 28 days inside Discovery) is the strongest
   pre-screen available.** Confirmed positive cases: PACK_ONLY iters 8, 9 (~1.5-5x transfer); PACK_PLUS_EXA
   iter 1 chain (~0.7x transfer, still positive). Negative case: PACK_ONLY iter 7 (calm-origin width
   conditioning did not transfer). Trust the OOS pre-screen over in-sample Discovery diagnostics.
6. **In-sample Discovery diagnostics LIE in both directions:** (a) widening levers look bad in-sample while
   the OOS/holdout median RISES (iter 8); (b) CENTERING levers that deviate from the pooled-OLS q50 are
   in-sample-negative by construction (the OLS q50 minimises in-sample loss) while being OOS/holdout-POSITIVE
   (PACK_PLUS_EXA iter 1: chain in-sample -0.0009, holdout +0.0010/+0.0006). A lever that is OOS-positive but
   in-sample-negative CANNOT pass the screen alone — stack it with a confirmed in-sample-positive lever.

## HARD guardrails (all measured; cite the falsifying iteration to remove any of them)
- **Bound every output.** Clip per-series z to a small multiple of pooled z AND clip final log-quantiles to a
  sane band around the last value (log q within +/- 1 of log last). (PACK_ONLY iter 2 explosion.)
- **Never touch the LOWER quantile with a "max".** (PACK_ONLY iter 3: q10 breach 10.1% -> 10.9%.) Any
  upper-tail widening experiment must leave z10 byte-identical to the champion.
- **Recent-window residuals do NOT estimate forward tails** (falsified: PACK_ONLY iters 2, 3, 7). The
  champion's full-window in-sample pooled z is honest in expectation. State/conditioned width levers are dead;
  unconditional smoothing (ensemble averaging) is the one width lever that WORKS (PACK_ONLY 8, 9).
- **Fixed-factor upper-tail (q90-only) widening is dead in ALL forms** (PACK_ONLY 3, 5 for h7; PACK_PLUS_EXA
  iter 1 analysis for q90-only at every horizon/factor 1.05-1.20: OOS-negative at every origin). The breach
  rate above 10% does not imply a width lever.
- **Validation-selected parameter changes degenerate to the champion** (PACK_ONLY iter 4). If a lever needs a
  gain it must change something unconditionally or its selection rule must be shown to fire on Discovery.
- **Holdout coverage must stay >= 0.74** (Gate-3 band floor). Champion 0.7514; the iter-1 stack sits at 0.7500
  (in-band). Coverage-neutral-or-widening changes only.
- **Momentum-beta shrinkage/damping is dead** (PACK_ONLY 4, 7): the pooled OLS beta is the out-of-sample
  optimum in both selection and unconditional forms. Per-instance-type hierarchy dead (PACK_ONLY 5).
- **In-sample mean residuals do NOT persist forward** (PACK_ONLY 5, 7). Do not "fix" bias with in-sample drift.
- **Per-series conditional width rules are dead** (PACK_ONLY 7). Only pooled/unconditional rules compete.
- **Cross-collector pull (kappa > 0) and cross-AZ consensus: falsified** (PACK_ONLY 1). Dead levers.
- **Ensemble with a drift/persistence member is dead** (PACK_ONLY 5); ensemble across perturbed
  hyperparameters of the SAME model is CONFIRMED-POSITIVE (PACK_ONLY 8, 9). Member set {7,14,28} is the
  optimum size; richer sets are OOS-negative (PACK_ONLY 9). Weighted blends ~ equal to plain average.
- **Exogenous market-median momentum from daily_statistics is dead** (PACK_ONLY 6): lagged and noisier than
  the per-AZ price. Long-lag momentum (14/30d) dead (PACK_ONLY 6). Regime-B spread floor dead (PACK_ONLY 6).
- **aggregator_medians and index_prints are degenerate for this pack** (PACK_ONLY 7). Dead as signals.
- **Regime-C q50 centering variants are dead** (PACK_ONLY 7). Train-day ensemble members {270,540} dead
  (PACK_ONLY 8). avgbeta is a wash — the ensemble gain lives in averaging the z/vol scale, not the point.
- Pool parameters across series where per-series histories are short (< 60 days).
- Point-in-time only: use rows dated on or before the origin; never hard-code dates or per-origin constants.
- Code: numpy/pandas only, vectorised, < 20 s per origin; guard empty histories and NaNs; always return
  every target series; enforce q10 <= q50 <= q90 and positivity. In analysis code, double-check lag indexing
  against a hand-computed example before trusting results (PACK_PLUS_EXA iter 1: a t+1-vs-t-1 indexing bug
  produced skill -1e7 and looked like a falsification until caught).

## Current champion state (PACK_ONLY iteration 8, PROMOTED)
Discovery pooled 0.4462, median 0.2831, coverage 0.7947. Holdout pooled 0.3088, median 0.2745, coverage
0.7514, horizons 0.4030/0.3369/0.2801. Gate margins all positive (gate1 +0.0092 vs its own predecessor).
Regimes: A AWS spot = pooled momentum OLS (train 540) + local-vol conformal with quantiles averaged across
vol_wins {7,14,28} at h3/h7 (h1 single-window + 0.95 tail narrowing); B rate cards = empirical change
quantiles + stickiness pin + IQR width; C marketplace/non-card = trailing-60-day change quantiles, kappa = 0.

## CONFIRMED-POSITIVE levers (all MEASURED on Holdout; do not re-measure, stack)
1. **h1 tail narrowing 0.95** (PACK_ONLY 5): holdout h1 +0.0047. In the champion.
2. **Regime-C recent-window change quantiles** (PACK_ONLY 6): holdout ~+0.0005 pooled. In the champion.
3. **Vol-window ensemble {7,14,28} at h3/h7** (PACK_ONLY 8, PROMOTED): holdout h3 +0.0124, h7 +0.0133.
4. **h1 vol-window ensemble + 0.90 re-center** (PACK_ONLY 9; re-measured in PACK_PLUS_EXA 1): holdout
   h1 +0.0134, pooled +0.0013, coverage 0.7500 in-band. NOT in the champion (sub-threshold alone).
5. **Chained (iterated 1-day) q50 at h3/h7** (PACK_PLUS_EXA 1): g[0]=last log value, g[j+1]=g[j]+beta1.(d1,d3,d7)
   with actual history for negative j, inside the ensemble members. Holdout h3 +0.0010, h7 +0.0006, pooled
   +0.0017 alone (chain-only part of the +0.0020 stack). In-sample Discovery NEGATIVE (-0.0009) — the direct
   pooled-OLS q50 is in-sample-optimal by construction; the chain is OOS-positive (5/5 origins) and
   holdout-positive. NOT in the champion.
6. **The full stack (4 + 5) measured on Holdout (PACK_PLUS_EXA iter 1, REVERTED on Gate 1 only, shortfall
   0.001):** pooled 0.3108 (+0.0020), median 0.2717 (-0.0029, Gate 2 pass), coverage 0.7500 (in-band),
   h1 0.4164 / h3 0.3379 / h7 0.2807. Gates 2/3/4 pass with margin. **The stack is +0.001 short of Gate 1.**
   Stacking confirmed levers adds up cleanly (iterations 6, 8, PACK_PLUS_EXA 1).

## Ranked untried directions (the lab decides stopping; these are the next hypotheses)
The stack of all confirmed levers is +0.0020 vs the +0.003 Gate-1 threshold: **any new confirmed lever worth
>= +0.001 pooled completes the stack.** Ranked by expected value:
1. **exogenous_signals (EXA web search — the stage's untried capability):** point-in-time search for AWS spot
   price change announcements / capacity crunches to shift the regime-A q50 or widen bands on flagged series.
   Must be pooled/unconditional in application (per-series conditional rules are dead), point-in-time cached,
   and shown to fire on Discovery before submission. If positive, stack with levers 4+5.
2. **exogenous_signals (pack tables): hardware_fundamentals / capability_ladder level anchoring for the
   marketplace and non-AWS hyperscaler q50** (bias_rel -0.05/-0.06 at h3/h7). Small slice (~4% of loss) but
   the bias is real and persistent; check the ladder maps to the benchmark gpu_models first.
3. **horizon_structure: chained variance accumulation** — the chain's z is still the direct-h residual
  quantile; a chained-residual z (residuals of the chained point forecast) is the natural completion of the
  confirmed chain lever. Measure OOS first; blends of chain/direct z were not tested.
4. **robustness: log-band clip (last log value +/- 1.0) bundled into the next candidate.** Pure safety,
   expected NO_OP on scored cells; bundle, never submit alone.
5. **cross_series: pooled cross-AZ residual correlation for the worst sticky series** (p5/p5e use1-az1/az4
   coverage 0.97, skill -0.7): their loss is WIDTH on calm sticky zones; only a pooled/unconditional
   treatment could work — low prior, measure OOS first.

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
