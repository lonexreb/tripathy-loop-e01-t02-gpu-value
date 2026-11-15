#!/usr/bin/env bash
# One-command reproduction from the unchanged pack: tests, baseline slice (F_0), improver cycle 1 (I_0 -> F_1), submission
# validation and the leakage refusal check. Needs only the pack and .venv (duckdb, pandas, numpy). Writes research/out, research/lineage.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python -I -B"
S="$ROOT/gpu-price-pack-v0.1.1-handover/starter"; O="$ROOT/research/out"
echo "== 0. pack integrity"; (cd "$ROOT/gpu-price-pack-v0.1.1-handover" && shasum -a 256 -c MANIFEST.sha256 | grep -v ': OK$' || true) | sed 's/^/MISMATCH /'
echo "== 1. tests"; $PY "$ROOT/research/test_scientist.py"; $PY "$ROOT/research/paired.py" --selftest
echo "== 2. baseline slice (F_0)"; $PY "$ROOT/research/run_baselines.py" | grep -vE '^\+ '
echo "== 3. improver cycle 1 (I_0)"; $PY "$ROOT/research/run_cycle1.py" | grep -vE '^(conformal_trailing|repricing_mixture): '
echo "== 4. leakage refusal check on the head submission (expected exit 2)"
set +e; $PY "$S/scorer.py" --forecast "$O/submission_head_2026-11-06.csv" --truth "$O/series_daily.csv" --history "$O/series_daily.csv" \
  --frozen-cutoff 2026-10-30 --out "$O/refusal_check.json" 2>&1 | tail -1; rc=${PIPESTATUS[0]}; set -e
echo "refusal exit code: $rc"; [ "$rc" -eq 2 ] || { echo "refusal check FAILED"; exit 1; }
echo "== 5. hashes"; shasum -a 256 "$O/submission_head_2026-11-06.csv" "$O/submission_F0_drift_2026-11-06.csv" "$ROOT/research/lineage/"*.json | sed "s#$ROOT/##"
