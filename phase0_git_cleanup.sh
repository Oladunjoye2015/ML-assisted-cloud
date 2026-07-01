#!/usr/bin/env bash
# Phase 0 git hygiene — run once in your local terminal, then review and commit.
# Safe & idempotent: only touches the git index. Your working files are NOT deleted.
set -euo pipefail
cd "$(dirname "$0")"

# 0. Clear any stale lock files left by an interrupted git process.
#    (A prior automated session may have left these; removing empty locks is safe
#     as long as no other git process is currently running in this repo.)
for lk in .git/index.lock .git/HEAD.lock .git/refs/heads/*.lock; do
  [ -e "$lk" ] && rm -f "$lk" && echo "removed stale lock: $lk"
done

echo "Untracking OS cruft and unused candidate models (files stay on disk)..."

# 1. Stop tracking .DS_Store anywhere.
git ls-files -z | grep -zZ '\.DS_Store$' | xargs -0r git rm --cached --quiet || true

# 2. Stop tracking the never-loaded candidate models (~430 MB).
#    Runtime only uses models/<PAIR>/best_model.pkl + metadata.
git ls-files -z 'models/*/candidate_models/*' | xargs -0r git rm --cached --quiet || true

echo
echo "Done. Review the staged deletions:"
echo "    git status --short | grep '^D'"
echo
echo "Then commit when you're happy:"
echo "    git commit -m 'Phase 0: untrack .DS_Store and unused candidate models; pin deps; add docs'"
echo
echo "Nothing was committed for you. Working-tree files are untouched."
