# Frozen research protocol — Tripathy Loop 1, Track 2 Direction A

Frozen 2026-11-14 before any comparative result beyond the official starter demo was inspected.
Edits after this date go in the change log at the bottom with a reason. The scientist (`F_g`, `I_g`) may not edit this file,
`origins.json`, `adapted_series.txt`, `paired.py`, or anything inside the pack.

## 1. Targets

- Adapted series: the 7 ids in `adapted_series.txt` (headline tier, H100/A100, daily truth). Instrument classes:
  `hyperscaler_spot` (afhubbard, 2 series), `marketplace_ask` (gpurentalprices, 1), `neocloud_rate_card` (gpurentalprices, 4).
  Each is a per-collector median of posted prices in USD per GPU-hour. Never pooled across collectors, never chained with AWS spot.
- Every other headline cell at the official origin is filled with the disclosed `drift` fallback (`submission_F0_drift_2026-11-06.csv`).
- Horizons 1, 3, 7. Horizon 30 and `aws_spot.*` are not Sunday-scored and are out of scope.

## 2. Origins and blocks (`origins.json`)

| block | origins | role |
|---|---|---|
| development | 2026-09-01 .. 09-30, daily | the improver may read per-cell errors here |
| checkpoint | 2026-10-01 .. 10-15, daily | acceptance test for candidates; errors unseen before a proposal |
| protected | 2026-10-16 .. 10-30, daily | scored once per final lineage member; common-start improver comparison |
| official | 2026-11-06 | submission; truth withheld until 2026-11-13 |

Daily origins overlap, so every uncertainty statement is a block bootstrap over origins (`paired.py`), never a cell-level test.

## 3. Scoring (fixed, outside the scientist)

- Starter `scorer.py` unchanged, backtest mode for blocks, `--frozen-cutoff 2026-11-06` for the submission.
- Cell loss: mean pinball at q10/q50/q90. Headline comparison: skill vs `last_value`; practical benchmark: paired mean pinball
  difference vs `drift` (starter `baselines.py --method drift --min-samples 20`, no lookback), 90% block-bootstrap CI.
- Coverage target 0.8 inside [q10, q90]; reported, not optimised.

## 4. Acceptance rule for a candidate F_{g+1}

Accepted iff, on the checkpoint block, paired mean pinball difference vs its parent is negative with the 90% CI excluding zero,
AND the development-block difference is not positive. Rejected candidates are kept in `lineage/` with `status: rejected` and the
same scores. Nothing is ever re-tested on the protected block to rescue it.

## 5. Budgets (per cycle, counted together: retrieval, diagnosis, candidate generation, debugging, evaluation, selection)

- Candidates: at most 2 per improver per cycle. Evaluations: each candidate scored once on development and once on checkpoint.
- Wall clock: 30 minutes per improver per cycle. Model: one fixed model id for every improver run, recorded in the lineage entry.
- External retrieval: at most 10 queries per cycle, each logged (section 7). The no-Exa control gets zero.

## 6. Access rules and admissibility

- As-of rule: any input, feature, benchmark, metadata or external document is admissible for origin `o` only if its publication
  date and its effective date are both ≤ `o`. Index prints use their own day basis (`DATA.md` §8); never align by naive date.
- Do-not-splice rule (`DATA.md` §3): rate cards, marketplace asks, spot, indices, resale pointers and launch prices stay separate
  instruments. They may be features for each other only as separate, as-of-filtered columns, never a merged price curve.
- Prohibited sources (`DATA.md` §9), as data or features: Vast.ai, RunPod, eBay-derived data, Reddit, auction sites, gpulist.ai,
  cloud-gpus.com, the SkyPilot catalog, the dstack bucket, and the Ornn / Silicon Data / SemiAnalysis dashboards (a cited headline
  print with attribution is allowed; storing or recomputing them is not). Anything admitted must be public, free, equally
  accessible, carry a source URL and a licence that was read, and travel with its `source_id`, `source_url`, `licence_id`.
- No PubMed-style shortcuts here either: no private data, no paid feeds, no scraping where terms bar it.

## 7. Evidence record (one JSON per retained external item, `research/evidence/`)

`query, url, title, published_date, effective_date, passage, retrieved_utc, sha256, admissible (bool + rule), used_by (lineage id),
decision_affected`. Reproduction reads only this frozen bundle; live retrieval is discovery-only.

## 8. Lineage artifact (`research/lineage/<id>.json`)

`id, generation, parent, kind (forecaster|improver), status (accepted|rejected|control), improver (id that proposed it),
method (exact executable description or path), params, budget_used, scores {block: {...}}, paired_vs_parent, evidence_ids, created_utc`.
`F_0` = starter drift. Controls carry `status: control`: `official_baselines`, `frozen_F0`, `fixed_improver_I0`, `recursive_I1`, `no_exa_I0`.

## 9. The RSI claim

Supported only if `I_1` (a child of `I_0` accepted under §4 on the checkpoint block) actually executes, proposes within budget, and
its selected descendant beats the descendant `I_0` selects from the same `F*`, same history through 2026-10-15, same model id,
same evidence rules and same budget, on the protected block, paired, with the CI excluding zero. A better final forecast alone is
reported as a forecasting result, not as RSI. Negative outcomes are written up with the same detail as positive ones.

## 10. Hypotheses queued for cycle 1 (each with its rejection condition)

1. Repricing mixture for stepwise posted prices: q50 stays at the last value, interval width from the empirical frequency and size of
   repricing events in the trailing window. Rejected if checkpoint paired diff vs drift is not negative with CI excluding zero.
2. Provider-composition diagnostic: flag days where `n_providers` changed; measure whether drift errors concentrate there. Diagnostic
   only; rejected as a feature if it does not change the selected candidate's checkpoint score.
3. Event features with publication and effective dates before each origin (NVIDIA newsroom, Epoch hardware data, BLS PPI). Rejected
   if the as-of-filtered feature adds nothing on checkpoint.

## Change log

- 2026-11-14: frozen.
