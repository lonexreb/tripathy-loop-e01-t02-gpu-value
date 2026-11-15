# Linear issue (paste-ready) — existing project, no new project

**Project (reuse):** https://linear.app/tripathy/project/tripathy-loop-e01-track-02-kalshi-gpu-forecaster-4a9aac262fa8/overview
Project name stays as is; its description gets one line: "Kalshi scope superseded 2026-11-14 by Track 2 Direction A, see ADR 0002."

**Title:** E01 · Track 2 Direction A — frozen protocol, baseline slice, cycle 1 (I_0 → F_1), cycle 2 common-start test

**Priority:** Urgent (P1) · **Estimate:** 5 pts · **Labels:** `track-2`, `direction-a`, `forecasting`, `rsi`, `research`, `supersedes-kalshi`
**Due:** 2026-11-15 (Sunday scoring) · **Repo:** https://github.com/lonexreb/tripathy-loop-e01-t02-gpu-value, branch `e01/track2-direction-a`
**Slack:** channel `C0CHNFW0DB7` (workspace `T0CHM33JRDH`)

## Description

Lane E01 moves from the provisional Kalshi replay scope to the Pacific Track 2 brief: q10/q50/q90 forecasts in USD per GPU-hour
for posted-rental series, origin 2026-11-06, horizons 1/3/7, mean pinball loss, skill vs `last_value`, practical benchmark `drift`.
The Kalshi history (commits `ac039bf`..`8189e37`, `docs/superseded/kalshi/`) is preserved as superseded, not deleted.

Research contribution: bounded recursive self-improvement, testing both the forecaster `F_g` and the improver `I_g` under a protocol
frozen before results were inspected (`research/PROTOCOL.md`, `research/origins.json`).

## Done so far (2026-11-14)

- Pack `v0.1.1-handover` verified (zip SHA-256, 672/672 manifest). Starter demo reproduced.
- Protocol frozen: 7 adapted H100/A100 headline series; blocks development 09-01..09-30, checkpoint 10-01..10-15, protected 10-16..10-30; budgets; as-of and prohibited-source rules (DATA.md §9).
- Baseline slice: drift beats last_value on every block with the 90% CI excluding zero, but 80% intervals cover only 0.59–0.67.
- Cycle 1, offline rule-based `I_0`: `repricing_mixture` rejected (checkpoint diff +0.000124, CI crosses 0); `conformal_trailing` window 45 accepted as `F_1` (checkpoint diff −0.000957, CI [−0.001539, −0.000315]; pooled skill +0.119 vs +0.089).
- `research/reproduce.sh`: one command from the unchanged pack, no dependency on `demo_out`; submission hashes deterministic; frozen validation exit 0, refusal check exit 2.

## Acceptance criteria (remaining)

- [ ] Cycle 2: `I_1` derived from `I_0` through the fixed evaluator; if accepted it must itself generate and select the next descendant.
- [ ] Common-start test: `I_0` vs `I_1` from the same `F*`, history through 2026-10-15, same model id and budget; descendants compared on protected cells, paired, 90% CI.
- [ ] Controls scored once on protected: official baselines, frozen `F_0`, fixed improver, recursive scientist, no-Exa scientist.
- [ ] Evidence bundle for every retained external item (query, URL, title, publication date, effective date, passage, retrieval time, hash, admissibility, decision affected).
- [ ] Findings with calibration and coverage analysis; negative results kept with the same detail as positive ones.
- [ ] Separate, explicitly conditional 24/36-month rental and resale scenarios.
- [x] Repository consolidated locally; Kalshi history preserved; remote renamed in place to the GPU-value scope.

## Out of scope

Kalshi replay, trading simulation, Brier scoring (superseded); AWS spot and horizon 30 (not Sunday-scored); Vast.ai, RunPod, eBay,
Reddit, auction sites, gpulist.ai, cloud-gpus.com, SkyPilot catalog, dstack bucket, Ornn / Silicon Data / SemiAnalysis dashboards as data.

## Blocked on

- No `EXA_API_KEY` / LLM key in the environment: Exa-enabled and model-driven improver arms wait; offline arms proceed.
- Remote writes (push, PR, tags) and external posts need explicit approval.

## Sub-issues

1. Cycle 2 `I_1` + common-start harness  2. Protected-block control scoring  3. Evidence bundle + no-Exa ablation
4. Findings + reproduction  5. 24/36-month scenario memo  6. Repo supersession commits (ADR 0002, README, project.yaml)
