#!/usr/bin/env bash
# Blocks commits that contain obvious credential patterns.
# pre-commit passes the staged filenames as "$@".
set -uo pipefail

PATTERN='(sk-[A-Za-z0-9_-]{20,}|sk-proj-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{34}|ghp_[A-Za-z0-9]{36}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)'
found=0

for f in "$@"; do
  [ -f "$f" ] || continue
  if grep -REn "$PATTERN" "$f" >/dev/null 2>&1; then
    echo "Possible secret in: $f"
    grep -REn "$PATTERN" "$f" | cut -c1-90 | sed 's/$/.../'
    found=1
  fi
done

if [ "$found" -ne 0 ]; then
  echo ""
  echo "Commit blocked. Move the value to .env (git-ignored) and read it with os.environ."
  exit 1
fi
