# Claude Code Lane 01

You own one isolated idea in the Tripathy Loop.

## Session protocol

- Read `project.yaml` and `experiment/manifest.json` first.
- Refuse to run an undefined experiment; complete the start gate instead.
- Plan around a falsifiable hypothesis and a fixed baseline.
- Use subagents for independent research, implementation, evaluation and adversarial review only when their contracts are explicit.
- Verify every subagent result before integration.
- Keep provider adapters behind interfaces; keep evaluation deterministic where possible.
- Update `artifacts/session-summary.json` after each meaningful iteration.
- Report metrics, latency, cost, failures and evidence—not confidence theater.
- Link every update to the Git SHA, Linear issue and run ID.
- Never include credentials in prompts, logs, artifacts, issues or Slack.
