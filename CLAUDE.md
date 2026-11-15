# Claude Code Lane E01 — Track 02 GPU Value

You own one isolated idea in the Tripathy Loop.

## Session protocol

- Read `project.yaml`, `experiment/manifest.json`, `PLAN.md`, and `research/PROTOCOL.md` first.
- Treat `research/PROTOCOL.md`, `research/origins.json`, `research/adapted_series.txt`, and `research/paired.py` as frozen evaluator inputs.
- Enforce time causality: no source, price, feature, or policy update may use information published after the simulated forecast origin.
- Plan around a falsifiable hypothesis and a fixed baseline.
- Never splice rate cards, marketplace asks, AWS spot, indices, resale pointers, or launch prices into one price curve.
- Preserve rejected candidates and negative results in `research/lineage/`.
- Use subagents only when their contracts are explicit and the user or repository instructions authorize delegation.
- Verify every subagent result before integration.
- Keep provider adapters behind interfaces; keep evaluation deterministic where possible.
- Update `artifacts/session-summary.json` after each meaningful iteration.
- Report metrics, latency, cost, failures and evidence—not confidence theater.
- Link every update to the Git SHA, Linear issue and run ID.
- Never include credentials in prompts, logs, artifacts, issues or Slack.
