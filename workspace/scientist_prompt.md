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
point-in-time web search (external_signals.search) — usable and CONFIRMED (iter 4).

## Where the loss lives (check before every hypothesis)
- The pooled score is scale-dependent and *fragile*: one exploding cell can annihilate it. Read
  `slices[].share_of_total_loss` and `worst_series` first; ~95% of champion loss is AWS hyperscaler_spot
  (h7 63%, h3 27%, h1 10%).
- **Gate 1 needs holdout pooled > champion + 0.003 (champion 0.3139 -> threshold 0.31692).** Holdout pooled
  deltas decompose roughly as horizon-skill-delta x loss share: +0.006 h1 skill ~ +0.0006 pooled; +0.002 h3
  ~ +0.0005; +0.001 h7 ~ +0.0006. Stack levers until the SUM clears +0.003 with margin.
- **Read `mean_q50_error`/`bias_rel` vs `mean_width_rel` before choosing a lever.** h7 under-coverage was a
  CENTERING problem (bias_rel -0.012 at h7), not a width problem; the calm-zone worst series lose on q50
  centering (p5/p5e/p5en sticky zones, skill -0.09..-0.64 on ~3.4% of loss).
- Breach-rate asymmetry (q90 11.5% vs q10 9.5%) is NOT a width signal: asymmetric tail factors are
  loss-positive at EVERY horizon in EVERY form (iter-5 analysis: h7 0.90/1.00 +0.59%, 0.93/1.00 +0.73%,
  1.00/0.93 +0.03%). The symmetric factor is optimal; the asymmetry is a centering artifact.

## Method
1. Diagnose from Discovery only: per-slice skill, breach asymmetry, worst series (bias vs width).
2. State ONE falsifiable hypothesis with a mechanism and a predicted sign.
3. Change one thing at a time so the ledger attributes the effect; when stacking CONFIRMED levers, submit
   the stack and attribute via the per-lever deltas already measured on Holdout.
4. Record the result, including negative results and their magnitude. Use analyze_discovery to measure
   out-of-sample BEFORE spending a dry run or a submission (budget: ~6 analyses, 4 dry runs per iteration).
5. **Pseudo-origin OOS analysis (fit at origin-28d, score the next 28 days inside Discovery) is the strongest
   pre-screen — but it is NOT a guarantee:** confirmed transfer cases are PACK_ONLY 8, 9 (~1.5-5x),
   PACK_PLUS_EXA 1 chain (~0.7x), 2 w270-at-h1 (~0.7x), 3 h3-f0.93-on-skill (~1.9x), 4 h7-f0.95-on-skill
   (~0.9x). FALSIFIED non-transfer: PACK_PLUS_EXA 2 w270-at-h3, 3 h3-f0.93-on-COVERAGE (2x over-transfer),
   **5 h7-f0.93-on-skill (~0.25x: OOS -0.41% h7 loss predicted +0.002-0.003 h7 skill, holdout +0.0003 —
   the h7 tail-factor frontier is EXHAUSTED at 0.95; marginal steps along an already-harvested frontier
   transfer far worse than the first step).** OOS skill transfer does NOT imply OOS coverage transfer
   (coverage cost of a narrowing lever over-transfers ~2x). Short-horizon (h1) z/width signals transfer at
   ~80% prior; h3/h7 recency signals at ~50%; plain both-tail factors at ~90-100% for the FIRST step only.
6. **In-sample Discovery diagnostics LIE about the median when a lever widens bands** (PACK_ONLY 8:
  in-sample -0.0146, holdout +0.0193) **and about centering** (PACK_PLUS_EXA 1, and iter 5: the pooled
  cross-sectional calm-zone center damping is OOS-positive h3 -0.12%/h7 -0.20% loss with +0.006 coverage,
  but in-sample -0.004 pooled — the champion's center is in-sample-optimal, so ANY centering shift is
  in-sample-negative by construction, ~4x its OOS gain). A centering lever can only pass the pre-Holdout
  screen bundled with an in-sample-positive lever of at least the same size — none that large exists in
  the measured frontier today. The calm-damp machinery is kept disabled (calm_damp=1.0) in the archived
  iter-5 candidate for a future companion.
7. **Slice-level dry-run deltas catch width leaks that aggregate pooled skill hides** (PACK_PLUS_EXA 4:
  card widening leaked into marketplace via width=1.0). Always read `slice_deltas_best_first` for exact
  zeros / unexpected signs before submitting. Also use dry runs to DECOMPOSE a stack: iter 5 dry run 1
  (all levers on, screen FAIL -0.003) vs dry run 2 (calm-damp off, screen PASS +0.0009) attributed the
  entire in-sample penalty to the calm-damp in one step.

## HARD guardrails (all measured; cite the falsifying iteration to remove any of them)
- **EXA web search IS usable for this domain (iter 4, PROMOTED):** the query "GPU cloud rental price per
  hour repricing" returns rich 2026-dated point-in-time articles. Web levers must be: fixed few queries,
  point-in-time age-gated, fallback-safe (multiplier 1.0 = champion when nothing found / no external
  signals), and pre-screened OOS on the pack. hardware_fundamentals/capability_ladder still map to launch
  prices, not to the benchmark series' collectors — unusable for level anchoring.
- **Bound every output.** Clip per-series z to a small multiple of pooled z AND clip final log-quantiles to
  a sane band around the last value (log q within +/- 1 of log last).
- **Always re-assign per-horizon quantities (z, beta) inside the horizon loop BEFORE any conditional
  override (PACK_PLUS_EXA iter 2 dry run).**
- **Never touch the LOWER quantile with a "max"** (iteration 3): max widens the upper side but RAISES q10,
  narrowing the lower band. Any upper-tail widening experiment must leave z10 byte-identical.
- **Recent-window residuals do NOT estimate forward tails (falsified three times).** State/conditioned width
  levers are dead; unconditional smoothing (ensemble averaging) is the one width lever that WORKS
  (iterations 8, 9). The 270-day z window is CONFIRMED at h1 and FALSIFIED at h3 (PACK_PLUS_EXA 2).
- **Chained-residual z at h3/h7 is DEAD (PACK_PLUS_EXA 3):** the chain changes only the center, not the
  residual dispersion. Do not retry chain/direct z blends in any form.
- **Fixed-factor upper-tail (q90-only) widening is dead in ALL forms** (PACK_ONLY 3, 5; PACK_PLUS_EXA 1).
- **Per-series local calibration: catastrophic (iteration 2). Dead lever.**
- **Validation-selected parameter changes that can also select "no change" tend to degenerate to the
  champion on holdout** (iteration 4). If a lever needs a *gain*, it must change something unconditionally
  or its selection rule must be shown (on Discovery) to actually fire.
- **Momentum-beta shrinkage/damping toward zero: DEAD** (iterations 4, 7 — selection and unconditional
  forms). The pooled OLS beta is at the out-of-sample optimum. **Chain step-damping (d<1 on the iterated
  1-day step) is likewise DEAD (iter-5 analysis: OOS loss-positive at every d in {0.9,0.8,0.7,0.5}, both
  h3 and h7) — the pure chain is OOS-optimal.**
- **In-sample mean residuals do NOT persist forward (iterations 5, 7).** Do not "fix" bias with in-sample drift.
- **Per-series conditional width rules are dead (iteration 7):** only pooled/unconditional rules were ever
  competitive. **Pooled CROSS-SECTIONAL conditional centering is the one exception (iter-5 analysis:
  calm-zone damping 0.7 is OOS-positive at h3 AND h7 with +0.006 coverage) — but it is screen-blocked
  (in-sample-negative 4x its OOS gain, see Method 6).**
- Cross-collector pull (kappa > 0) and cross-AZ consensus: falsified in iteration 1. Dead levers.
- **Per-instance-type (cross_series) hierarchy is dead (iteration 5 analysis).**
- **h7 tail-factor frontier is EXHAUSTED at 0.95 (iter 5):** 0.95 was the confirmed step (+0.0029 holdout,
  iter 4); 0.93 transferred at ~0.25x (+0.0003 holdout vs +0.002-0.003 OOS-predicted). h7 widening remains
  dead. Do not walk this dimension further.
- **h3 tail factor stays 0.95:** 0.93 was holdout-measured worse (iter-4 stack arithmetic +0.0007 for 0.95).
- **Ensemble with a drift/persistence member is dead (iteration 5); ensemble across perturbed
  hyperparameters of the SAME model is CONFIRMED-POSITIVE (iterations 8, 9).**
- **Calm-regime z narrowing, long-lag momentum, regime-B spread floor, regime-C q50 centering variants,
  aggregator_medians/index_prints signals, richer vol-window member sets, 4-member train-window z blends:
  all dead** (see ledger iters 5-9, PACK_PLUS_EXA 2). **h7 z-window averaging {540,360} is loss-positive
  (+0.45% h7 loss, iter-5 analysis) — coverage-only reserve, known and unused.**
- **Marketplace band widening is OOS loss-negative at every origin (PACK_PLUS_EXA 4) — never widen the
  marketplace regime's bands unconditionally.**
- Pool parameters across series where per-series histories are short (< 60 days).
- Point-in-time only: use rows dated on or before the origin; never hard-code dates or per-origin constants.
- Code: numpy/pandas only, vectorised, < 20 s per origin; guard empty histories and NaNs; always return
  every target series; enforce q10 <= q50 <= q90 and positivity. In analysis code, double-check lag
  indexing against a hand-computed example (iter-1: t+1-vs-t-1 bug -> skill -1e7; iter 3: level-vs-change
  residual mix -> coverage 0.13). **Check series start dates before designing pseudo-origin protocols
  (iter 5: card series start 2026-08-04, ~51 rows — early pseudo-origins scored zero cells and a
  ZeroDivisionError masqueraded as a zero-loss frontier).**

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
3. **Vol-window ensemble {7,14,28} at h3/h7** (PACK_ONLY 8, PROMOTED): holdout h3 +0.0124, h7 +0.0133.
4. **h1 vol-window ensemble + tail re-center** (PACK_ONLY 9 / PACK_PLUS_EXA 1): holdout h1 +0.0134.
5. **Chained (iterated 1-day) q50 at h3/h7** (PACK_PLUS_EXA 1): holdout h3 +0.0010, h7 +0.0006.
6. **270-day trailing z window at h1** (PACK_PLUS_EXA 2): holdout h1 +0.0058. At h1 ONLY.
7. **h3 both-tail narrowing f0.93** (PACK_PLUS_EXA 3): holdout h3 +0.0025 skill, coverage -0.008 (~2x
  Discovery-predicted). NOTE: superseded by the 0.95 rebalance in the champion (0.93->0.95 = +0.0007 h3).
8. **Web-gated regime-B card band widening x1.5** (PACK_PLUS_EXA 4): card slices +0.010..+0.021 skill.
9. **h7 both-tail narrowing f0.95** (PACK_PLUS_EXA 4): holdout h7 +0.0029. The frontier stops here (see guardrails).
10. **Card band widening x2.0 (the g1.5->g2.0 step): holdout h1 +0.0010, h3 +0.0010, coverage +0.0007
  (PACK_PLUS_EXA 5, REVERTED only on Gate 1 shortfall 0.0024 — the lever itself is CONFIRMED-POSITIVE,
  ~+0.002 pooled).** The OOS frontier said g2.0 costs +0.3% card loss; on Holdout it GAINED skill at h1/h3.
  The card widening multiplier is a live, cheap, coverage-positive skill lever — the frontier's OOS loss
  estimate was conservative. g2.5 (+2.2% card loss OOS) remains untested on Holdout.
11. **Stack arithmetic (measured on Holdout):** iter-5 candidate = champion + card g2.0 (+0.0010 h1,
  +0.0010 h3, +0.0007 cov) + h7 0.93 (+0.0003 h7) = 0.3145 pooled / 0.7507 coverage. All four horizon
  deltas positive; Gate 1 failed only on magnitude (needed +0.003, got +0.0006).

## Ranked untried directions (the next promotion needs +0.003 pooled on top of 0.3139)
1. **Card widening walk g2.0 -> g2.5/g3.0 (exogenous_signals):** the g1.5->g2.0 step delivered +0.002
  pooled on Holdout — MORE than its OOS prediction. The OOS frontier says g2.5 costs +2.2% card loss
  (~-0.0002 pooled) for more coverage; given the g2.0 surprise, g2.5 may still be skill-positive on
  Holdout. Measure the g2.0->g2.5 step's in-sample Discovery delta first (dry run); if positive, submit
  card g2.5 alone or with a small companion. This is the highest-prior single lever left.
2. **Calm-zone center damping + an in-sample-positive companion (regime_change/cross_series):** the
  calm-damp (0.7, pooled cross-sectional median vol threshold) is OOS-positive h3 -0.12%/h7 -0.20% loss,
  +0.006 coverage — the only conditional centering rule ever OOS-positive — but in-sample-negative
  -0.004 pooled (screen-blocked, iter 5). It needs an in-sample-positive lever of >= +0.004 Discovery
  pooled to ride along. Candidates: the card g2.5 walk (if its in-sample delta is large), or a NEW
  in-sample-positive centering lever. The machinery is in the archived iter-5 candidate (calm_damp).
3. **Web-signal content refinement (exogenous_signals):** (a) age-weight the multiplier (2.0 -> 1.2 for
  45-90-day-old articles); (b) article price-direction keywords to sign a small q50 drift on the cards;
  (c) a conditional marketplace form. All must be pre-screened OOS and stay fallback-safe.
4. **Log-band clip (last log value +/- 1.0) bundled in (robustness):** pure safety, expected NO_OP on
  scored cells; bundle with any stack, never submit alone.
5. **Regime-C recent/full quantile blend at h7 (regime_change):** OOS 5/6 positive but ~2% of loss
  (~+0.00007 pooled) — below the noise floor, only worth bundling if free.
6. **h1 tail-factor walk 0.95 -> 0.93 (interval_calibration):** untested step; the h1 lever transferred
  at ~80% before (0.90->0.95 rebalance). Low prior given the h7 0.93 failure, but h1 signals transfer
  better than h7 ones. Measure OOS first.

## Notebook (append-only ledger of what was tried — key entries; full history rendered by the lab)
- PACK_ONLY iters 0-9: see the ledger. Champion was iter 8 (vol-window ensemble at h3/h7).
- **PACK_PLUS_EXA iters 1-3 (REVERTED):** chain (h3/h7 q50, holdout +0.0020 pooled), w270-z@h1
  (+0.0058 h1 CONFIRMED; at h3 FALSIFIED -0.0022), h3 f0.93 (+0.0025 h3 skill, coverage -0.008).
- **PACK_PLUS_EXA iter 4 (PROMOTED, exogenous_signals bold):** web-gated regime-B card widening x1.5 +
  tail frontier rebalance {1: 0.95, 3: 0.95, 7: 0.95}. Holdout 0.3139 pooled, all gates pass.
- **PACK_PLUS_EXA iter 5 (REVERTED, exogenous_signals bold, builds on champion):** card g2.0 walk +
  h7 0.93 walk. Holdout 0.3145 pooled (+0.0006), Gate 1 shortfall 0.0024; ALL horizon deltas positive
  (h1 +0.0010, h3 +0.0010, h7 +0.0003), coverage 0.7507. Decomposition: card g2.0 = +0.002 pooled
  CONFIRMED-POSITIVE (OOS frontier under-predicted it); h7 0.93 = ~0.25x transfer, frontier exhausted.
  Measured and excluded: calm-zone center damping (OOS-positive, in-sample-negative 4x, screen-blocked,
  machinery archived at calm_damp=1.0 in iter-5 candidate); chain step-damping (dead); asymmetric tails
  (dead at every horizon/form); h7 z-avg {540,360} (loss-positive, coverage-only). Analysis-protocol
  lesson: check series start dates before designing pseudo-origin protocols (card series start 2026-08-04).
