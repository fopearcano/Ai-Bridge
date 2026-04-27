#!/usr/bin/env bash
# Launch the Ai-Bridge_Houdini Qt UI from a fresh checkout.
# Requires the [gui] extra: `pip install -e .[gui]` (or `pip install PySide6`).
# Usage: ./run_ui.sh [args...]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${AIBRIDGE_PYTHON:-${PYTHON:-python3}}"
export PYTHONPATH="${DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" -m aibridge_houdini --ui qt "$@"
