#!/bin/bash
# Run the independent ASL fingerspelling recognizer.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x ".venv-alphabet/bin/python" ]; then
    PYTHON=".venv-alphabet/bin/python"
elif [ -x ".venv-repair/bin/python" ]; then
    PYTHON=".venv-repair/bin/python"
else
    echo "ERROR: Create the isolated .venv-alphabet environment; see docs/ALPHABET_MODE.md." >&2
    exit 1
fi

"$PYTHON" tools/run_alphabet_demo.py "$@"
