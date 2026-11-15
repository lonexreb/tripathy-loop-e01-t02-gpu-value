# Scientist Playbook — GPU cloud rental price forecaster (stage PACK_ONLY)

<!-- LAB-PROTECTED:BEGIN (owned by the lab; rewrites cannot remove or edit this block) -->
## Lab rules (protected)
- The lab, not you, decides when research stops. "Accept/freeze the champion" is never a valid plan: every
  iteration must test a real, falsifiable change. A string of reverts means the explored families are exhausted,
  not that the problem is solved; move to an untried family (see the experiment registry) or make a bolder change.
- Every hypothesis is tagged with one family and a novelty (incremental | bold). Do not spend more than two
  consecutive experiments in one family without a promotion; after two reverts in a row, the next one is bold.
- Near-misses are archived with their code, diff and diagnostics (inspect_candidate) and can be built on, but a
  near-miss of the same model class is a local move. A small Holdout gain is not "confirmed": Holdout has 5 origins
  and per-candidate deltas of +-0.002 are noise. Do not stack levers by their Holdout deltas.
- Every candidate is tagged with a model_class (see the lab's list). When the lab directive says MODEL-CLASS
  EXPLORATION, write a forecaster of a different class for the AWS regime from scratch (write_candidate), compare
  several classes on Discovery (save_variant / load_variant) and submit the best one.
- One failed implementation falsifies that implementation, not the idea. Only mark a lever or class "dead" after
  two independent implementations fail (an implementation bug, e.g. an exploding quantile, falsifies nothing).
- A candidate whose Holdout aggregates equal the champion's (NO_OP) changed nothing that is scored: its
  selection rule never fired. Make changes that act unconditionally or show on Discovery that they fire.
- Guardrails are only removed with evidence: cite the iteration that falsified them in the rationale.
- Keep this playbook a distilled strategy (heuristics, guardrails, ranked ideas). The full experiment history
  is rendered by the lab from the ledger; do not copy the notebook into the playbook.
<!-- LAB-PROTECTED:END -->

## Mission
Beat the champion on the four Holdout gates (pooled skill, breadth, calibrated 80% band, no bad horizon)
with forecasts that would also be honest at the frozen 2026-11-06 origin.

## Current champion state (PACK_ONLY iteration 8, PROMOTED)
Discovery pooled 0.4462, median 0.2831, coverage 0.7947. Holdout pooled 0.3088, median 0.2745, coverage
0.7514, horizons 0.4030/0.3369/0.2801. Gate-1 threshold for the next promotion: holdout pooled > 0.3118.
Champion model class: pooled_linear_momentum (regime A: pooled OLS momentum + vol-window ensemble
{7,14,28} at h3/h7 + 0.95 h1 tail narrowing + vol-scaled conformal z; regimes B/C empirical change quantiles).

## Where the loss lives (check before every hypothesis)
- ~95% of champion loss is AWS hyperscaler_spot (h7 ~63%, h3 ~27%, h1 ~10%). One exploding cell annihilates
  the pooled score: read `slices[].share_of_total_loss` and `worst_series` first.
- Read `mean_q50_error`/`bias_rel` vs `mean_width_rel` before choosing a lever: most "width" problems are
  centering problems (iter-10: the GBM's under-coverage was centering-driven; after the center fix the same
  machinery covered 0.789 vs 0.761).

## Method
1. Diagnose from Discovery only: per-slice skill, breach asymmetry, worst series (bias vs width).
2. State ONE falsifiable hypothesis with a mechanism and a predicted sign.
3. Change one thing at a time so the ledger attributes the effect; when stacking CONFIRMED levers, submit
   the stack and attribute via per-lever deltas already measured on Holdout.
4. Record the result, including negative results and their magnitude. Use analyze_discovery to measure
   out-of-sample BEFORE spending a dry run or a submission (budget: ~6 analyses, 8 dry runs per iteration).
5. **Dry-run budget is precious: a crashed dry run still consumes one.** Iter 10 lost its last dry run to a
   trivial unpack bug and then the Holdout submission CRASHED on the same bug class (a 4-tuple unpacked as
   5) — the submission budget was consumed by a RUNTIME_ERROR and nothing was evaluated. Before ANY dry run
   or submission: re-read the changed lines, check every tuple unpack against the actual tuple arity, and
   prefer indexing (`t[0]`, `t[1]`) over starred/unpack patterns in dict comprehensions.
6. **In-sample Discovery diagnostics LIE about the median when a lever widens bands** (PACK_ONLY 8:
  in-sample -0.0146, holdout +0.0193) **and about centering** (the champion's center is in-sample-optimal by
  construction, so ANY centering shift is in-sample-negative, ~4x its OOS gain). A centering lever can only
  pass the pre-Holdout screen bundled with an in-sample-positive lever of at least the same size.
7. Slice-level dry-run deltas catch width leaks that aggregate pooled skill hides (e.g. card widening leaking
  into marketplace via width=1.0). Always read `slice_deltas_best_first` for exact zeros / unexpected signs
   before submitting. Also use dry runs to DECOMPOSE a stack lever-by-lever.
8. **Submission gate (mechanical, measured twice: g001_frozen DRY_RUN_ERROR, iter_010 RUNTIME_ERROR):**
   the EXACT code committed to Holdout must have completed at least one successful dry run with ZERO edits
   after that dry run. Any edit after the last successful dry run — however trivial, even a comment or a
   renamed variable — requires one fresh dry run before commit_forecast. A crashed dry run must be fixed and
   re-run before any submission. Reading the diff (rule 5) is necessary but not sufficient: the changed lines
   must have actually executed at least once under the byte-identical submitted code. If the dry-run budget
   cannot afford this final verification, do not edit after the last dry run — submit the code that already
   ran clean.

## HARD guardrails (all measured; cite the falsifying iteration to remove any of them)
- **EXA web search IS usable for this domain (PACK_PLUS_EXA iter 4, PROMOTED):** the query "GPU cloud rental
  price per hour repricing" returns rich 2026-dated point-in-time articles. Web levers must be: fixed few
  queries, point-in-time age-gated, fallback-safe (multiplier 1.0 = champion when nothing found / no external
  signals), and pre-screened OOS on the pack. hardware_fundamentals/capability_ladder still map to launch
  prices, not to the benchmark series' collectors — unusable for level anchoring. (PACK_ONLY stage has no
  external_signals; relevant when the stage re-enables EXA.)
- **Bound every output.** Clip per-series z to a small multiple of pooled z AND clip final log-quantiles to
  a sane band around the last value (log q within +/- 1 of log last). NOTE: `lv` in this codebase is ALREADY
  a log value — never apply np.log to it (iter-10 analog bug: clip bounds collapsed to exp(-12.8) = 2.7e-6,
  REJECTED_SANITY).
- **Always re-assign per-horizon quantities (z, beta) inside the horizon loop BEFORE any conditional override
  (PACK_PLUS_EXA iter 2 dry run).**
- **Never touch the LOWER quantile with a "max"** (PACK_ONLY iteration 3): max widens the upper side but
  RAISES q10, narrowing the lower band. Any upper-tail widening experiment must leave z10 byte-identical.
- **Recent-window residuals do NOT estimate forward tails (falsified three times).** State/conditioned width
  levers are dead; unconditional smoothing (ensemble averaging) is the one width lever that WORKS
  (PACK_ONLY iterations 8, 9). The 270-day z window is CONFIRMED at h1 and FALSIFIED at h3 (PACK_PLUS_EXA 2).
- **Chained-residual z at h3/h7 is DEAD (PACK_PLUS_EXA 3):** the chain changes only the center, not the
  residual dispersion. Do not retry chain/direct z blends in any form.
- **Fixed-factor upper-tail (q90-only) widening is dead in ALL forms** (PACK_ONLY 3, 5; PACK_PLUS_EXA 1).
- **Validation-selected parameter changes that can also select "no change" tend to degenerate to the
  champion on holdout** (PACK_ONLY iteration 4). If a lever needs a *gain*, it must change something
  unconditionally or its selection rule must be shown (on Discovery) to actually fire.
- **Per-instance-type (cross_series) hierarchy is dead (PACK_PLUS_EXA iter 5 analysis).**
- **h7 tail-factor frontier is EXHAUSTED at 0.95 (PACK_PLUS_EXA iter 5):** 0.95 was the confirmed step
  (+0.0029 holdout, iter 4); 0.93 transferred at ~0.25x. h7 widening remains dead. **h3 tail factor stays
  0.95:** 0.93 was holdout-measured worse (iter-4 stack arithmetic +0.0007 for 0.95).
- **Ensemble with a drift/persistence member is dead (PACK_PLUS_EXA iteration 5); ensemble across perturbed
  hyperparameters of the SAME model is CONFIRMED-POSITIVE (PACK_ONLY iterations 8, 9).**
- **Calm-regime z narrowing, long-lag momentum, regime-B spread floor, regime-C q50 centering variants,
  aggregator_medians/index_prints signals, richer vol-window member sets, 4-member train-window z blends:
  all dead** (see ledger iters 5-9, PACK_PLUS_EXA 2). **h7 z-window averaging {540,360} is loss-positive
  (+0.45% h7 loss) — coverage-only reserve, known and unused.**
- **Marketplace band widening is OOS loss-negative at every origin (PACK_PLUS_EXA 4) — never widen the
  marketplace regime's bands unconditionally.**
- **Per-series local calibration: catastrophic (PACK_ONLY iteration 2). Dead lever.**
- **Momentum-beta shrinkage/damping toward zero: DEAD** (PACK_ONLY iterations 4, 7 — selection and
  unconditional forms). The pooled OLS beta is at the out-of-sample optimum. (Iter-10's "shrink toward
  persistence" tests were invalidated by a center-inflation bug — shrink toward the LINEAR member's q50 is
  still untested, see ranked directions.)
- **In-sample mean residuals do NOT persist forward (PACK_ONLY iterations 5, 7).** Do not "fix" bias with
  in-sample drift.
- **Per-series conditional width rules are dead (PACK_ONLY iteration 7):** only pooled/unconditional rules
  were ever competitive.
- **Cross-collector pull (kappa > 0) and cross-AZ consensus: falsified (PACK_ONLY iteration 1). Dead levers.**
- Pool parameters across series where per-series histories are short (< 60 days).
- Point-in-time only: use rows dated on or before the origin; never hard-code dates or per-origin constants.
- **Center arithmetic in log-change space:** q50_new = shrink * q50_old scales the center's deviation from
  persistence. `q50_old + shrink * q50_old` DOUBLES the center at shrink=1.0 (iter-10 bug: three dry runs
  measured center inflation, not shrink/widening). Write the arithmetic as a one-line comment and hand-check
  it on a concrete number before running.
- Code: numpy/pandas/sklearn fine, vectorised, < 20 s per origin (GBM member: ~7 s/origin, fits the budget);
  guard empty histories and NaNs; always return every target series; enforce q10 <= q50 <= q90 and positivity.
  In analysis code, double-check lag indexing against a hand-computed example (iter-1: t+1-vs-t-1 bug ->
  skill -1e7; iter 3: level-vs-change residual mix -> coverage 0.13). **Check series start dates before
  designing pseudo-origin protocols** (PACK_PLUS_EXA iter 5: card series start 2026-08-04, ~51 rows — early
  pseudo-origins scored zero cells and a ZeroDivisionError masqueraded as a zero-loss frontier).

## MODEL-CLASS EXPLORATION results (iteration 10, PACK_ONLY)
- **gradient_boosting (HistGradientBoostingRegressor, loss='quantile', features = 1/3/7-day log changes +
  7/14/28-day local vol, pooled over AWS series, 540-day train): Discovery pooled 0.4856 vs champion 0.4462
  (+0.039, 10/11 origins, h1 +0.070 / h3 +0.036 / h7 +0.036 skill).** The largest single-model Discovery gain
  measured in this lab: NONLINEARITY of the state->change map was the binding constraint, not the band
  machinery. BUT median skill 0.2438 vs 0.2831 (-0.039, tree q50 drifts on calm sticky zones) — Gate-2 risk.
- **analog_nonparametric (kernel-weighted quantiles of standardized historical paths): Discovery pooled 0.4264
  (-0.020, screen FAIL) despite median +0.020. NEGATIVE class result in this form** — standardized-path
  resampling smooths the momentum signal the pooled loss lives on. One implementation; a second independent
  implementation (e.g. unstandardized analogs, per-series bandwidth) would be needed to call the class dead.
- **model_combination (50/50 log-space quantile average GBM x pooled-linear-momentum): the iter-10 submission
  CRASHED (unpack bug) and was never evaluated.** The FIXED code is in the workspace / archived candidate
  iter_010; expected Discovery profile between the members on both axes (pooled 0.45-0.48, median 0.24-0.28).
- GBM under-coverage (q90 breach 14-17% at h3/h7) is centering-driven: symmetric tail widening after the
  center fix (gbm_wide3: 0.4808 pooled / 0.7886 cov) was slightly WORSE than the plain GBM — widening is not
  needed once the center is correct.
- Untried classes remaining: quantile_regression, state_space, volatility_model, regime_switching,
  hierarchical_pooling. Untried families: cross_series, exogenous_signals, robustness.

## Ranked untried directions (next promotion needs holdout pooled > 0.3088 + 0.003)
1. **RESUBMIT the fixed model_combination (iter-10 candidate, highest priority):** dry-run it first (the
   unpack fix has not been executed once), then submit. Both members clear the Discovery screen; the blend is
   expected between them. If Gate 2 (median >= 0.2695) fails while Gate 1 passes, re-weight w_gbm 0.5 -> 0.3
   or shrink the tree q50 toward the LINEAR q50 (not toward persistence — that was never actually tested).
2. **GBM member refinement (gradient_boosting family):** the plain GBM's +0.039 pooled is the biggest lever
   ever measured; its median deficit is the only blocker. Options: (a) blend weight sweep on Discovery
   (w_gbm in {0.3, 0.5, 0.7}); (b) add the champion's vol-scaled conformal z as a THIRD member; (c) richer
   features (lagged vol ratios, day-since-last-change for sticky zones).
3. **quantile_regression class (untried):** direct linear/spline quantile regression per horizon — the
   natural non-tree alternative to the GBM member; compare on Discovery against the GBM.
4. **state_space / volatility_model classes (untried):** local-level Kalman or EWMA-vol band with a simple
   center — cheap to build, likely a blend member rather than a solo winner.
5. **cross_series family (untried):** hierarchical pooling across AZs of the same instance type — note the
   per-instance-type hierarchy was falsified in PACK_PLUS_EXA iter 5; only pooled/cross-sectional forms have
   ever worked in this lab.

## Notebook (append-only ledger of what was tried — key entries; full history rendered by the lab)
- PACK_ONLY iters 0-9: champion is iter 8 (vol-window ensemble at h3/h7, holdout 0.3088).
- **PACK_ONLY iter 10 (MODEL-CLASS EXPLORATION, RUNTIME_ERROR — submission crashed on a tuple-unpack bug,
  Holdout never evaluated, champion unchanged):** built and dry-ran three classes from scratch.
  gradient_boosting 0.4856 pooled (+0.039, screen PASS, median -0.039 Gate-2 risk); analog_nonparametric
  0.4264 (screen FAIL, class-negative in this form, after fixing a double-log clip bug);
  model_combination GBM x linear 50/50 — chosen for submission, crashed. Three intermediate dry runs
  (gbm_wide/wide2/wide3) were invalidated by a center-inflation arithmetic bug (`q50 + shrink*q50` doubles
  the center); the corrected pure-widening test showed widening is unnecessary. The fixed combination
  forecaster is in the workspace, ready to dry-run and resubmit next iteration.
