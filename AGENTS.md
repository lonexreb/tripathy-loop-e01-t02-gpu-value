# AI Scientist for GPU rental price forecasting: project notes

- Python env: `.venv` (uv, Python 3.12). Always run with `PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1` and never write
  into `gpu-price-pack-v0.1.1-handover/` (symlink to the read-only pack; MANIFEST.sha256 must keep verifying).
- Tests (deterministic, no LLM/network): `.venv/bin/python -m pytest -q -p no:cacheprovider tests`
- Full offline reproduction: `bash reproduce.sh` (about 45 s); add `--live N` for N live Stage-1 iterations.
- RSI loop: `python -m lab.rsi_controller run --stage pack_only|pack_plus_exa --driver modal|api|strands|scripted --iterations N`;
  `fork-stage2`, `status`, `ablation` subcommands. Stage 2 (Exa) needs EXA_API_KEY.
- LLM backend: Strands `OpenAIModel` against the Modal endpoint
  `https://ananthbashyam1--ep-glm-5-3-server.us-west.modal.direct` (zai-org/GLM-5.3) with Modal proxy auth
  (`MODAL_PROXY_TOKEN_ID` / `MODAL_PROXY_TOKEN_SECRET` in the gitignored `.env`). No Anthropic SDK.
- `lab/` is immutable to the agent (hash-checked); the agent owns `workspace/`.
