#!/bin/bash
# Run the unified one-camera word/alphabet mode selector.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=".venv-repair/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Required word/UI environment is missing: $PYTHON" >&2
    exit 1
fi

exec "$PYTHON" tools/run_sign2text_integrated.py "$@"
