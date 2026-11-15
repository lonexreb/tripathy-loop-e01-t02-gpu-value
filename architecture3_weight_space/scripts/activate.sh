#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Run: source scripts/activate.sh" >&2
  exit 2
fi

if [[ -n "${ZSH_VERSION:-}" ]]; then
  ARCH3_ACTIVATE_SCRIPT="${(%):-%x}"
else
  ARCH3_ACTIVATE_SCRIPT="${BASH_SOURCE[0]}"
fi

ARCH3_WORKSPACE="$(cd "$(dirname "$ARCH3_ACTIVATE_SCRIPT")/.." && pwd)"

if [[ ! -f "$ARCH3_WORKSPACE/.env" ]]; then
  echo "Missing $ARCH3_WORKSPACE/.env" >&2
  return 2
fi

if [[ ! -f "$ARCH3_WORKSPACE/.venv/bin/activate" ]]; then
  echo "Missing .venv; run 'uv sync --extra dev' first" >&2
  return 2
fi

set -a
source "$ARCH3_WORKSPACE/.env"
set +a
source "$ARCH3_WORKSPACE/.venv/bin/activate"
export ARCH3_WORKSPACE
