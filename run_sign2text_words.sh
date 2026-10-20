#!/bin/bash
# Run the independent pretrained isolated-word recognizer.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x ".venv-words/bin/python" ]; then
    PYTHON=".venv-words/bin/python"
elif [ -x ".venv-repair/bin/python" ]; then
    PYTHON=".venv-repair/bin/python"
else
    echo "ERROR: No compatible word-model environment found; see docs/WORD_MODE.md." >&2
    exit 1
fi

"$PYTHON" tools/run_word_demo.py "$@"
