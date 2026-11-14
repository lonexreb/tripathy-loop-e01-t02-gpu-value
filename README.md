# Tripathy Loop E01 - Track 02: Kalshi GPU Forecaster

Tripathy Loop lane `E01`, assigned provisionally to Pacific Track 02: Recursive Self-Improvement. The candidate system is an evidence-grounded H100 forecaster evaluated by leakage-safe replay of settled Kalshi markets.

The scope is not implementation-ready. Independent Claude and ChatGPT reviews must resolve data availability, cutoff integrity, baseline timing, the primary metric, budget and stop condition before the start gate passes. See `docs/PROJECT_BRIEF.md` and `docs/CROSS_REVIEW_PROMPT.md`.

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
- Slack broadcasts linked updates to `#tl-e01-t02-kalshi`.
- Claude Code owns this lane and delegates bounded subagent tasks.
- Modal runs isolated compute when the selected idea requires it.
- Vercel hosts only an idea-specific UI, if needed.

Read `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/EXPERIMENT_PROTOCOL.md` and `docs/SECURITY.md` before implementation.
