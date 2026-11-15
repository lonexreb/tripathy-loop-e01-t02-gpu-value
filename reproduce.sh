#!/usr/bin/env bash
# One-command offline reproduction: pack integrity, official starter benchmark, deterministic tests, bit-exact replay
# of every recorded champion, ablation table, frozen 2026-11-06 submission (validated by starter/scorer.py),
# valuation and findings report. No LLM and no network calls (Exa is read from outputs/exa_cache.jsonl only).
# Usage: bash reproduce.sh            (offline)
#        bash reproduce.sh --live N   (also runs N live RSI iterations of Stage 1 with the Modal GLM endpoint; needs MODAL_PROXY_TOKEN_ID/SECRET in .env)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PACK=gpu-price-pack-v0.1.1-handover
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 LAB_EXA_CACHE_ONLY=1
if [ ! -x .venv/bin/python ]; then
  uv venv -q --python 3.12 .venv && uv pip install -q --python .venv/bin/python -r requirements.txt
fi
PY=.venv/bin/python
t0=$(date +%s)

echo "== 0. clean derived state (the research record: ledger, champions, notebook, exa cache, is kept)"
rm -rf outputs/cache outputs/starter_demo outputs/ablation_summary.csv outputs/findings_report.md outputs/valuation.json \
       "outputs/submission_2026-11-06.csv" "outputs/submission_2026-11-06_scorer_report.json"

echo "== 1. pack integrity (MANIFEST.sha256) and the official starter demo"
(cd "$PACK" && shasum -a 256 -c MANIFEST.sha256 --quiet) && echo "manifest OK"
PYTHON=$PY bash "$PACK/starter/run_demo.sh" "$PACK" outputs/starter_demo 2026-10-24 > outputs/starter_demo.log 2>&1
grep -E "refusal check exit code|frozen_cutoff.*report_frozen_drift|backtest.*report_drift" outputs/starter_demo.log

echo "== 2. series cache + deterministic tests (Tasks 2-4)"
$PY -c "from lab.sandbox import PointInTimeSandbox as S; S().ensure_series_cache()"
$PY -B -m pytest -q -p no:cacheprovider tests

if [ "${1:-}" = "--live" ]; then
  echo "== 2b. live RSI (Stage 1, Strands + Modal GLM endpoint)"
  $PY -B -m lab.rsi_controller run --stage pack_only --driver modal --iterations "${2:-3}"
fi

echo "== 3. bit-exact replay of the recorded champions + ablation summary"
$PY -B -m lab.rsi_controller ablation
echo "== 4. frozen submission at 2026-11-06 (validated by starter/scorer.py)"
$PY -B -m lab.submit
echo "== 5. valuation + findings report (with champion verification)"
$PY -B -m lab.report --verify
(cd "$PACK" && shasum -a 256 -c MANIFEST.sha256 --quiet) && echo "pack still unmodified"
echo "done in $(( $(date +%s) - t0 )) s"
