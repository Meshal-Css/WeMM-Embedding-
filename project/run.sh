#!/usr/bin/env bash
# Launch the interface with absolute paths — works from any directory.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/.venv/bin/python" "$ROOT/project/app.py"
