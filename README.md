# Tripathy Experiment 01

Tripathy Loop lane `01`. The research idea has not been assigned.

## Start gate

Do not build idea-specific functionality until `experiment/manifest.json` contains a falsifiable hypothesis, baseline, metric, budget and stop condition.

## Commands

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,modal]'
pytest
python -m tripathy_experiment.cli validate
python -m tripathy_experiment.cli run --dry-run
```

## Repository contract

- GitHub records code, review, test results and artifacts.
- Linear records work, ownership and decisions.
- Slack broadcasts linked updates to `#tripathy-exp-01`.
- Claude Code owns this lane and delegates bounded subagent tasks.
- Modal runs isolated compute when the selected idea requires it.
- Vercel hosts only an idea-specific UI, if needed.

Read `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/EXPERIMENT_PROTOCOL.md` and `docs/SECURITY.md` before implementation.
