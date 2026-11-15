# Architecture 3 — Qwen weight-space optimization

This isolated workspace implements the Architecture 3 extension for the existing
Tripathy Loop E01 Track 2 lane and Linear issue `TRI-33`.

The intended actor is the official post-trained `Qwen/Qwen3.5-0.8B`. Its base
weights remain frozen while supervised fine-tuning updates a LoRA adapter. A
closed critic proposes corrections, a dataset compiler verifies them, and the
existing fixed evaluator decides whether an adapter is accepted.

## Security contract

- Real credentials live only in `.env`, which is ignored by Git and permissioned
  for the current user.
- `.env.example` contains names and non-secret defaults only.
- Modal authentication stays in the Modal CLI profile at `~/.modal.toml`; Modal
  token values are not copied into this repository.
- Environment checks report only whether a credential is present, never its value.
- Checkpoints, cached weights, and run artifacts are ignored by Git.

`HF_TOKEN` is optional because the Qwen checkpoint is public. A critic credential
is required only after `ARCH3_CRITIC_PROVIDER` and `ARCH3_CRITIC_MODEL` are chosen.
`EXA_API_KEY` is required only for retrieval-enabled research episodes.

Configured services:

- OpenAI critic: `gpt-6.1-sol`, using the restricted key
  `tripathy-e01-arch3-critic` (model requests only; expires 2026-12-15).
- Exa retrieval: dedicated key `tripathy-e01-arch3-retrieval`.
- past.dev memory: `default-project`; the API key is stored only in the protected
  local `.env`.
- Modal compute: CLI profile `reach2shubhankar`; its token remains outside `.env`.

## Local setup

```bash
uv sync --extra dev
source scripts/activate.sh
arch3-env-check
arch3-preflight
pytest
```

The activation script loads this folder's `.env` into the current shell and
activates its `.venv`. The Python configuration loader also reads `.env`
directly, so commands remain deterministic when invoked through `uv run`.

`arch3-preflight` verifies the published pack ZIP checksum, every file in the
pack manifest, the frozen protocol/evaluator inputs, the pinned model settings,
the bounded remote execution authorization, and redacted credential presence. It writes the complete
result to `artifacts/preflight/latest.json`. Use `--skip-pack-files` only for a
quick development check; it is not the reproducibility acceptance command.

The controlling implementation scope is [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
Provider-independent schemas live in `tripathy_arch3.contracts`; each worker's
event stream is an append-only, hash-chained JSONL shard managed by
`tripathy_arch3.audit`.

## Local data-boundary acceptance

```bash
uv run arch3-phase2-acceptance
```

This builds a content-addressed actor bundle containing only the seven frozen
target series through the checkpoint cutoff, invokes the unchanged official
scorer through an evaluator-only truth handle, and compares every aggregate to
the frozen checkpoint drift report. The actor bundle contains one allowlisted
CSV and no full-truth path or evaluator artifact. When actor execution moves to
Modal, this bundle—not the repository or full pack—is the only data mount; OS
isolation and blocked networking are enforced at that later remote-execution
gate.

## GPU boundary

`modal_app.py` defines the future Modal GPU boundary, package versions, and a
Volume v2 checkpoint mount. Merely importing or testing this workspace does not
start a GPU or create a training run.

The user-authorized remote scope and hard call/cost ceilings are recorded in
`config/execution_authorization.json`. Git commit and push remain closed. Every
remote operation must consume a recorded allowance before it runs.

## Phase 3 evidence and memory acceptance

```bash
uv run arch3-phase3-acceptance
```

The live acceptance freezes exact provider requests and responses. Exa historical
content fails closed when Snapshot is unavailable; current content is never used
as a substitute. past.dev memory is isolated by a fixed audience and reader,
polled to completion, and admitted only when every recalled source maps to the
local lineage registry and was available by the requested perspective.

## Phase 4–5 execution

`modal_app.py` contains pinned L4 functions for model sanity and deterministic G0
actor inference, plus amended H100/B200 functions for LoRA-only training and
adapter inference with base-tensor hashes and save/reload logit equivalence.
`arch3-modal-sanity` must pass before
`arch3-phase4-episode`, which commits and scores the G0 forecast before making
the two fixed OpenAI teacher calls. Modal initially refused the L4 function
before container start because the account had no payment method; that exact
failure remains preserved under `artifacts/phase4/`. The payment gate was later
resolved, and the pinned L4 sanity and G0 episode passed. A subsequent
user-approved hardware amendment uses H100 with B200
fallback for the remaining training and adapter inference because L4 scheduling
latency exhausted the available time; the budget and scientific protocol remain
unchanged.

## Phase 6 admission and report

`arch3-phase6-admission` loads the immutable G1 adapter, asks it for one bounded
forecaster patch from development-only context, and evaluates that fixed patch
once on development and once on checkpoint. The decision is the unchanged
`research/PROTOCOL.md` rule: checkpoint difference below zero with the complete
90% block-bootstrap interval below zero, plus a non-positive development
difference. Rejected candidates are preserved and never evaluated on protected
truth. Only an accepted candidate receives the one protected-block evaluation.

`arch3-report` regenerates `artifacts/report/` from local artifacts and labels
missing remote work as `not_run`. It never converts an environment check, SFT
loss, or saved curriculum into a forecasting or RSI claim.

## Completed run

Run `g1-6ccddfe047031e9c` completed on an NVIDIA H100 80GB HBM3. The one-step
LoRA update changed adapter tensors, preserved the frozen base-tensor hash, and
reloaded with a maximum absolute logit difference of `0.0`. The child emitted a
valid forecaster patch, but the frozen admission gate rejected it: paired mean
pinball loss was worse than the parent on both development and checkpoint, with
the checkpoint 90% bootstrap interval entirely above zero. Protected truth was
therefore not scored, and no recursive-self-improvement claim is established.

The final local report is `artifacts/report/summary.json`. All 31 tests, Ruff,
and mypy pass. The recorded remote budget is `$0.674` estimated and exactly
`1200` reserved GPU-seconds.
