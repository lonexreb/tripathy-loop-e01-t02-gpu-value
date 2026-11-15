# Git migration record — E01 Track 2 GPU Value

Canonical local repository: `/Users/shubhankar/Documents/OpenAlex/type_classification/iterate-ai-hackathon/tripathy-loop1`

Canonical remote: `https://github.com/lonexreb/tripathy-loop-e01-t02-gpu-value`

Working branch: `e01/track2-direction-a`

## Preserved history

The original four E01 commits remain unchanged:

- `ac039bf` — establish Tripathy Loop baseline
- `623c5b1` — link Linear project and Slack lane
- `688eb42` — provisional Track 2 Kalshi scope
- `8189e37` — stable Slack lane alias

Local tag `kalshi-scope-superseded` points to `8189e37`. ADR 0002 explains the migration, and the former project brief, cross-review prompt, and manifest remain under `docs/superseded/kalshi/`.

## Consolidation

The existing Git repository became the only canonical working tree and was renamed locally to `tripathy-loop1`. The non-Git research workspace was merged into it. No second repository was initialized and no history was rewritten.

The immutable pack, source ZIP, virtual environment, starter-demo output, and `research/out/` remain local and ignored. Code, protocol, lineage, contract, checksum, attribution, and operational records are tracked.

## Verification gate

- `bash research/reproduce.sh`
- `python -m tripathy_experiment.cli validate`
- `pytest`
- `ruff check .`
- `mypy`
- secret-pattern scan and ignored-bulk audit

The migration branch is pushed only after every gate passes. The GitHub repository was renamed in place from `tripathy-loop-e01-t02-kalshi` to `tripathy-loop-e01-t02-gpu-value`; GitHub preserves redirects from the former URL.
