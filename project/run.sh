#!/usr/bin/env bash
# يشغّل الواجهة بمسارات مطلقة — يعمل من أي مجلد.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/.venv/bin/python" "$ROOT/project/app.py"
