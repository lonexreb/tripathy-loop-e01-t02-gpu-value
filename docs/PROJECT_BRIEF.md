# Provisional Project Brief

## Identity

- Machine name: `tripathy-loop-e01-t02-kalshi`
- Display name: `Tripathy Loop E01 - Track 02: Kalshi GPU Forecaster`
- Track: Pacific Track 02, Recursive Self-Improvement
- Stackable challenge: Exa Research Agents
- Decision state: candidate scope; start gate not passed

## Official requirements, normalized

The Track 02 system must:

1. act as an AI scientist forecasting GPU/H100 prices;
2. improve as it runs by learning from its earlier predictions and newly available evidence;
3. replay every available settled Kalshi H100 weekly and monthly market using only information available before that market closed;
4. compare its forecast error with the market's error;
5. produce a defensible view of H100 value two to three years out; and
6. demonstrate rigour, economic insight and reproducibility.

For the stacked Exa challenge, Exa must be central to evidence discovery. The output must also demonstrate research value, creativity and checkability.

## Adapted research statement

Build a time-causal, evidence-grounded AI forecaster that replays settled Kalshi H100 weekly and monthly markets using only evidence and market information available at each simulated forecast timestamp. For each market, the system must publish its forecast, the time-matched Kalshi baseline, a dated evidence ledger, assumptions and a machine-checkable score. After settlement, an auditable self-improvement loop may diagnose errors and revise the research or forecasting policy, but the revised policy can affect only later replay steps. The experiment will test whether these revisions reduce proper-scoring loss relative to the market without look-ahead leakage. The calibrated short-horizon results will then support a scenario-based, uncertainty-aware estimate of H100 value two to three years out.

## Provisional falsifiable thesis

An auditable `forecast -> score -> diagnose -> revise` loop using cutoff-bounded Exa research and explicit economic features will achieve lower out-of-sample Brier score than a contemporaneous, time-matched Kalshi market-implied baseline across the eligible settled H100 markets.

This thesis is provisional. It must be revised or killed if the historical market inventory, price history or publication cutoffs cannot support a fair walk-forward replay.

## Proposed evaluation contract

- Unit of evaluation: one eligible settled Kalshi H100 market.
- Replay order: chronological by simulated forecast timestamp.
- Primary metric: paired Brier-score difference, candidate minus market; lower is better.
- Secondary metrics: log loss, calibration error, coverage, latency, research cost and source-quality checks.
- Baseline: Kalshi-implied probability captured at the same timestamp as the candidate forecast.
- Leakage rule: no evidence, price, settlement fact or learned policy from the future may enter an earlier replay step.
- Uncertainty: report the per-market score table and a paired bootstrap interval; do not hide a small sample behind a mean.
- Checkable output: one JSON record per market containing cutoff, evidence provenance, forecast, baseline, settlement, scores, policy version, Git SHA and run ID.

## Recursive self-improvement, minimally defensible

Self-improvement means updating an explicit, versioned forecasting policy after observing an eligible prior result. Examples include changing query strategy, source weighting, feature selection, calibration or error-checking rules. It does not mean repeatedly prompting until a favorable backtest appears. Every policy change needs a reason, a diff and evaluation only on later replay steps.

## Economic model candidates

- accelerator supply and generation transitions;
- cloud rental prices and utilization;
- memory, networking, power and data-center constraints;
- export controls and geographic segmentation;
- hyperscaler and model-training demand;
- depreciation, resale markets and substitute accelerators.

These are candidate features, not accepted causal claims.

## Start-gate blockers

1. Enumerate the actual settled Kalshi H100 weekly and monthly markets.
2. Verify access to time-stamped historical market probabilities at a fair comparison time.
3. Verify that Exa can retrieve dated, pre-cutoff evidence without later-page contamination.
4. Fix the forecast timestamp policy, primary metric, sample inclusion rules, budget and stop condition.
5. Decide whether the sample size permits an improvement claim or only a reproducible case study.

## Kill and pivot rules

- Kill the market-beating claim if historical probabilities cannot be reconstructed at matched timestamps.
- Pivot to a reproducible evidence-and-calibration benchmark if there are too few eligible settled markets.
- Exclude a market when cutoff, settlement or baseline provenance cannot be independently checked.
- Do not use the two-to-three-year forecast as evidence that the backtest succeeded; it is a downstream scenario output.
