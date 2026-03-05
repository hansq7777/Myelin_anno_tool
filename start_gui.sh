#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_CMD=""
if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  PYTHON_CMD="$SCRIPT_DIR/.venv/bin/python"
elif [[ -x "$SCRIPT_DIR/venv/bin/python" ]]; then
  PYTHON_CMD="$SCRIPT_DIR/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "Python not found. Please install Python 3." >&2
  exit 1
fi

if [[ "${1:-}" == "--check" ]]; then
  echo "[Myelin_anno_tool] Python command: $PYTHON_CMD"
  exec "$PYTHON_CMD" --version
fi

echo "[Myelin_anno_tool] Starting GUI..."
exec "$PYTHON_CMD" -m zstack_anno
