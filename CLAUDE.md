# Claude Code Lane E01 - Track 02 Kalshi

You own one isolated idea in the Tripathy Loop.

## Session protocol

- Read `project.yaml` and `experiment/manifest.json` first.
- Read `docs/PROJECT_BRIEF.md` and the two independent model reviews before proposing an implementation plan.
- Refuse to run an undefined experiment; complete the start gate instead.
- Treat every field prefixed `PROVISIONAL:` as unresolved; do not silently promote it to an accepted design decision.
- Enforce time causality: no source, market price, feature or policy update may use information published after the simulated forecast timestamp.
- Plan around a falsifiable hypothesis and a fixed baseline.
- Use subagents for independent research, implementation, evaluation and adversarial review only when their contracts are explicit.
- Verify every subagent result before integration.
- Keep provider adapters behind interfaces; keep evaluation deterministic where possible.
- Update `artifacts/session-summary.json` after each meaningful iteration.
- Report metrics, latency, cost, failures and evidence—not confidence theater.
- Link every update to the Git SHA, Linear issue and run ID.
- Never include credentials in prompts, logs, artifacts, issues or Slack.
