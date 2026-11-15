# Tripathy Loop 1 — Track 2 execution plan

Status: contract acquired, data pack verified and unpacked, official starter workflow reproduced, frozen protocol implemented, and offline Cycle 1 completed on 2026-11-14.

## Execution lane

Reuse the existing E01 lane; do not create a parallel project or rewrite its history:

- GitHub: <https://github.com/lonexreb/tripathy-loop-e01-t02-gpu-value>
- Linear: <https://linear.app/tripathy/project/tripathy-loop-e01-track-02-gpu-value-4a9aac262fa8>
- Slack: <https://app.slack.com/client/T0CHM33JRDH/C0CHNFW0DB7> (workspace `T0CHM33JRDH`, channel `C0CHNFW0DB7`)

The four existing Kalshi-scoped commits remain unchanged and readable. ADR 0002 marks that scope as superseded by this Pacific Track 2 Direction A contract, and the former documents remain under `docs/superseded/kalshi/`. The GitHub repository was renamed in place, branch `e01/track2-direction-a` is pushed with PR #1, the existing Linear project was updated with issue `TRI-33`, and the migration result was posted to the existing Slack channel.

Current implementation checkpoint:

- `research/reproduce.sh` rebuilds the baseline and Cycle 1 directly from the immutable pack, without relying on `demo_out/`.
- `I_0` proposed exactly two bounded candidates under the frozen evaluator.
- `repricing_mixture` was rejected: checkpoint paired difference `+0.000124`; the 90% CI crosses zero.
- `conformal_trailing` with window 45 was accepted as `F_1`: checkpoint paired difference `-0.000957`, 90% CI `[-0.001539, -0.000315]`; pooled skill improved from `+0.089` to `+0.119` versus `last_value`.
- Cycle 1 candidate selection did not use the protected block; Cycle 2 must preserve the common-start comparison defined by the protocol.

## Governing decision

Use the five-page Pacific Track 2 brief as the working contract and pursue Direction A: self-improving GPU rental-price forecasts. The primary scored object is not a Kalshi probability. It is a set of `q10`, `q50`, and `q90` forecasts in USD per GPU-hour for the supplied series.

Kalshi, trading simulation, and binary Brier scoring are outside the core unless the organizers separately reinstate them. The research contribution remains bounded recursive self-improvement: test both the forecaster and the procedure that proposes forecasting improvements.

## Acquired inputs and integrity

Source URLs:

- Contract: <https://drive.google.com/file/d/1r-dIO5Ol6RYLmcHXRwEUFr1vIBpi5CiZ/view?pli=1>
- Data folder: <https://drive.google.com/drive/folders/14zWi964QlLh3mZeWGrBI85J31-SqfGHN>

Local inputs:

- Contract PDF: `source-downloads/Track 2 - Pacific - AI Scientist for GPU Value - Brief.pdf`
- Original archive: `source-downloads/gpu-price-pack-v0.1.1-handover_pack-only.zip`
- Download instructions: `source-downloads/READ ME FIRST.txt`
- Published archive checksum: `source-downloads/ZIP.sha256`
- Unpacked immutable data pack: `gpu-price-pack-v0.1.1-handover/`

Integrity checks completed:

- Published ZIP SHA-256: `f0b52af28c52545732208bcc4905facd9b941aa5779da683e1d9884c283b7c7c`
- Downloaded ZIP matches the published checksum.
- All 672 files named by the pack's `MANIFEST.sha256` match; mismatches: 0.
- Preserve the ZIP and extracted pack unchanged. Put every generated artifact outside the pack.

## Verified data contract

- Pack version: `v0.1.1-handover`.
- Information cutoff: 2026-11-06 23:59:59 UTC.
- The 2026-11-07 through 2026-11-13 scoring window is withheld.
- Rental values are posted prices, not transacted prices.
- `daily_statistics.csv` is the Sunday prediction-target source.
- AWS spot series are development-backtest series unless an authorized November truth collection exists.
- Resale data is pointer-only; there is no shipped public transaction-level resale dataset.
- Do not splice rate cards, marketplace asks, spot series, indices, resale pointers, or hardware release prices into one synthetic price curve.
- Do not add sources prohibited by `DATA.md` section 9, including Vast.ai, RunPod, eBay-derived data, Reddit, auction sites, gpulist.ai, cloud-gpus.com, the SkyPilot catalog, or the dstack bucket as data.

## Official scoring contract

- Forecast columns: `series_id, origin_date, horizon_days, q10, q50, q90`.
- Official final origin: 2026-11-06.
- Sunday-scored horizons: 1, 3, and 7 days.
- Metric: mean pinball loss across quantiles 0.1, 0.5, and 0.9.
- Headline comparison: skill relative to `last_value`; the practical baseline to beat is `drift` with calibrated intervals.
- Missing submitted cells are filled with the last-value baseline.
- Frozen-cutoff mode must reject forecasts or history that cross the origin boundary.

## Reproducibility checkpoint — completed

Environment:

- Isolated Python 3.12 environment: `.venv/`
- Starter dependencies installed from `gpu-price-pack-v0.1.1-handover/starter/requirements.txt`
- Official outputs: `demo_out/`

Command reproduced:

```bash
PATH="$PWD/.venv/bin:$PATH" \
  bash gpu-price-pack-v0.1.1-handover/starter/run_demo.sh \
  gpu-price-pack-v0.1.1-handover demo_out 2026-10-24
```

Observed results:

| Check | Result |
|---|---:|
| Full series build | 103,151 rows, 725 series, 97 headline |
| Truncated history | 95,648 rows, 658 series, 97 headline |
| Scored demo cells | 291 |
| `last_value` pooled skill | 0.000000 |
| `last_value_empirical` pooled skill | +0.207101 |
| `drift` pooled skill | +0.200409 |
| `seasonal_naive` pooled skill | -0.275699 |
| Leakage refusal | exit code 2, as required |
| 15-origin `drift` backtest pooled skill | +0.173426 |

The exact pack produces 725 full-file series, rather than relying on the earlier brief's illustrative catalog count. Treat the generated catalog as authoritative for implementation.

## Experimental plan

### Phase 0 — contract and integrity gate — completed

1. Acquire the brief and pack through the participant path.
2. Preserve the original archive.
3. Verify the published ZIP checksum and internal manifest.
4. Read `DATA.md`, `ATTRIBUTION.md`, the starter README, and the forecast schema.

### Phase 1 — immutable baseline harness — completed

1. Run the supplied series builder, baselines, sample submission, scorer, refusal check, and rolling backtest unchanged.
2. Keep the pack read-only in practice by writing all artifacts to sibling directories.
3. Treat `demo_out/series_catalog.csv` and the official scorer as the implementation contract.

### Phase 2 — freeze the research protocol

1. Predeclare adapted series: H100/A100 headline posted-rental series, with disclosed `drift` fallback for all other required cells.
2. Predeclare development, improvement-checkpoint, common-start, and protected origins before inspecting comparative results.
3. Apply as-of filtering to every input, including external evidence, benchmarks, metadata, and derived transforms.
4. Fix total research budgets for every comparison, counting retrieval, diagnosis, candidate generation, debugging, evaluation, and selection.
5. Freeze scorer, target definitions, exclusions, origin/horizon set, and access rules outside the editable scientist.

### Phase 3 — forecasting hypotheses

Run a small, auditable hypothesis set against `drift`:

1. No-change versus repricing mixture for stepwise posted prices.
2. Provider-composition diagnostics using like-for-like offers without replacing the official target.
3. Selective external-event features whose publication and effective dates precede each origin.

Each hypothesis must have a rejection condition and be evaluated on later, unexposed origins. Negative results are retained.

### Phase 4 — bounded recursive-improvement test

Represent the scientist as a forecaster `F_g` and improver `I_g`.

1. Cycle 1: `I_0` may propose at most two bounded candidates, including a forecasting change and/or a diagnostic or experiment-selection change.
2. Accept changes only through the fixed evaluator; retain rejected candidates and exact lineage.
3. Cycle 2: if `I_1` is accepted, it must actually generate and select the next descendant.
4. Common-start test: give `I_0` and `I_1` the same `F*`, history, model version, evidence rules, and total budget; compare their selected children on the same protected cells.
5. Required controls: official baselines, frozen forecasting method, fixed improver, recursive scientist, and no-Exa scientist.

The RSI claim is supported only if the changed improver executes and produces a better protected descendant under matched conditions. A better final forecast alone is insufficient.

### Phase 5 — evidence and provenance

Use the chain:

`error -> unresolved question -> evidence -> hypothesis -> executable change -> later behavior -> score`

For every retained external source, store the exact query, URL, title, publication date, event/effective date, relevant passage, retrieval time, content hash, admissibility decision, and downstream decision it affected. Reproduction uses a frozen evidence bundle; live retrieval is discovery-only.

### Phase 6 — final deliverables

1. Complete quantile forecast CSV for the official origin and scored horizons.
2. Official scorer reports with paired comparisons against `last_value` and `drift`.
3. Calibration and interval-coverage analysis.
4. Executable researcher lineage with accepted and rejected changes.
5. Common-start improver comparison and no-Exa ablation.
6. One-command reproduction from the unchanged pack and frozen evidence.
7. Separate, explicitly conditional 24- and 36-month rental and resale scenarios. Do not imply the short-horizon score validates those scenarios.

## Immediate next actions

1. Build a read-only adapter around `demo_out/series_catalog.csv` and select the exact H100/A100 headline series.
2. Freeze historical origins and protected blocks before model comparison.
3. Implement the fixed `drift` benchmark wrapper and paired pinball-loss report.
4. Implement the bounded `F_g`/`I_g` artifact and lineage format.
5. Run the first hypothesis cycle and retain all failures.
