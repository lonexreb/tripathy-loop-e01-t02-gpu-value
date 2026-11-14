# Independent Cross-Review Prompt

Copy the prompt below unchanged into both Claude and ChatGPT. Run it in fresh chats so neither reviewer anchors on the other's answer.

---

You are an independent research-design reviewer. Do not implement the system and do not assume the proposal is viable. Your job is to find fatal flaws, leakage, unfair comparisons and overclaiming before a hackathon team spends build time.

## Official Track 02 brief

- Build an AI scientist that forecasts GPU/H100 prices better than the market and estimates H100 value two to three years out.
- The forecaster should improve as it runs, learning from earlier predictions and newly available evidence.
- Benchmark requirement: replay every eligible settled Kalshi H100 weekly and monthly market using only data available before it closed, then compare forecast error with the market's.
- Track judging: rigour, economic insight and reproducibility.

## Stackable Exa challenge

- Exa must be central to a research agent that discovers information it did not already know and exposes real sources.
- Judging: Exa centrality, research value, creativity and checkable output.

## Candidate project statement

Build a time-causal, evidence-grounded AI forecaster that replays settled Kalshi H100 weekly and monthly markets using only evidence and market information available at each simulated forecast timestamp. For each market, publish the forecast, a time-matched Kalshi baseline, dated evidence, assumptions and machine-checkable scores. After settlement, run an auditable `forecast -> score -> diagnose -> revise policy` loop; a revised policy may affect only later chronological replay steps. Test whether this lowers proper-scoring loss relative to the market without look-ahead leakage. Use the calibrated short-horizon replay to support a scenario-based distribution for H100 value two to three years out.

## Provisional evaluation

- Primary metric: paired Brier-score difference versus Kalshi; lower is better.
- Secondary: log loss, calibration, latency, cost and source-quality checks.
- Baseline: market-implied probability captured at the exact same forecast timestamp.
- Output: one JSON record per market with cutoff, evidence provenance, forecast, baseline, settlement, scores, policy version, Git SHA and run ID.
- Exa role: retrieve time-bounded evidence and produce the auditable evidence ledger.

## Review tasks

1. Map the candidate to every explicit Track 02 and Exa criterion. Mark each as satisfied, partial, missing or unverifiable.
2. Audit data feasibility. State exactly what Kalshi market inventory, historical price snapshots, close times and settlement data are required. If browsing is available, verify availability with direct links; otherwise label this unverified.
3. Red-team temporal leakage, publication-date ambiguity, survivorship bias, baseline timing, market selection and sequential policy updates.
4. Decide whether Brier score is the correct primary metric. Propose a better metric only if you can explain why it is fairer and remains comparable with the market.
5. Define the smallest defensible form of recursive self-improvement. Separate real policy learning from prompt retrying, test-set tuning and narrative relabeling.
6. Judge whether Exa is genuinely central or merely attached. Specify a central Exa workflow and its checkable artifacts.
7. Assess whether the two-to-three-year H100 estimate is logically supported by the replay. Propose the narrowest honest bridge between short-horizon calibration and long-horizon scenarios.
8. Produce a ruthless hackathon scope: what must be built, what can be mocked, and what must be cut.
9. Identify the single most likely fatal blocker and a time-boxed feasibility test for it.
10. Rewrite the project statement and falsifiable hypothesis if needed.

## Required response format

1. Verdict: `GO`, `CONDITIONAL GO`, `PIVOT` or `KILL`
2. Criterion matrix
3. Verified facts versus assumptions
4. Fatal blockers and leakage risks
5. Recommended question, hypothesis, primary metric and baseline
6. Minimum viable recursive-improvement loop
7. Exa-central architecture and checkable outputs
8. Time-boxed feasibility gate
9. Build / mock / cut list
10. Final rewritten project statement
11. Five decisions the human team must make

Be adversarial, concise and explicit about uncertainty. Do not award credit for sophistication that cannot be demonstrated reproducibly.

---
