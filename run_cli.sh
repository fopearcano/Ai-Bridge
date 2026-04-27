#!/usr/bin/env bash
# Launch the Ai-Bridge_Houdini CLI from a fresh checkout.
# Bootstraps PYTHONPATH to ./src so `pip install -e .` is optional.
# Usage: ./run_cli.sh [args...]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${AIBRIDGE_PYTHON:-${PYTHON:-python3}}"
export PYTHONPATH="${DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" -m aibridge_houdini "$@"
