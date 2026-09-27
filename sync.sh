#!/usr/bin/env bash
# Sync a local Hermes skill to its GitHub repo (jgbarahbot/<skill>).
# Commits local changes, fast-forwards, pushes. Uses GITHUB_CONTENT_RW_TOKEN from ~/.hermes/.env.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL="$(basename "$DIR")"
REPO="jgbarahbot/$SKILL"
TOK="$(grep '^GITHUB_CONTENT_RW_TOKEN=' "$HOME/.hermes/.env" | head -n1 | cut -d= -f2- | tr -d '\r')"
[ -n "$TOK" ] || { echo "ERROR: GITHUB_CONTENT_RW_TOKEN not found in ~/.hermes/.env" >&2; exit 1; }
cd "$DIR"
git add -A
if ! git diff --cached --quiet; then
  git -c user.name="jgbarahbot" -c user.email="jgbarahbot@users.noreply.github.com" commit -m "sync: $SKILL $(date -u +%F)"
  echo "committed local changes"
fi
URL="https://jgbarahbot:${TOK}@github.com/${REPO}.git"
git fetch "$URL" main
if ! git merge --ff-only FETCH_HEAD >/dev/null 2>&1; then
  echo "ERROR: cannot fast-forward; resolve diverged history manually" >&2
  exit 1
fi
git push "$URL" main
echo "synced $SKILL -> $REPO (up to date)"
