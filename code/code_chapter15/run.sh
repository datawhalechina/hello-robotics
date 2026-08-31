#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OPENPI="${CHAPTER15_OPENPI_ROOT:-$ROOT/third_party/openpi}"
export CHAPTER15_OPENPI_ROOT="$OPENPI"
export PYTHONPATH="$ROOT:$OPENPI/src:$OPENPI/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$OPENPI/.venv/bin/python" "$@"
