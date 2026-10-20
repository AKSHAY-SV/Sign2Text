#!/bin/bash
# Run the pretrained ASL sign recognizer with webcam-based Holistic landmarks.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x ".venv-repair/bin/python3" ]; then
    PYTHON=".venv-repair/bin/python3"
elif [ -x ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
else
    echo "ERROR: No project virtual environment found (.venv-repair or .venv)." >&2
    exit 1
fi

"$PYTHON" launch_demo.py "$@"
