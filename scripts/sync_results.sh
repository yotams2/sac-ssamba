#!/bin/bash
# Commit whatever is new in the results worktree and push it to the `results` branch.
# Usage: scripts/sync_results.sh [commit message]
# Always exits 0, so a job is never marked failed because of it. A push that fails leaves the commit in place, and the
# next sync from any job or session pushes it, even if that sync has nothing of its own to commit.
ROOT=${SAC_RESULTS_ROOT:-$HOME/sac-ssamba-results}
MSG=${1:-results: sync}
LOCK_WAIT=${RESULTS_LOCK_WAIT:-300}

[ -e "$ROOT/.git" ] || { echo "[sync_results] no results worktree at $ROOT"; exit 0; }
cd "$ROOT" || exit 0

exec 9>"$ROOT/.sync.lock"
flock -w "$LOCK_WAIT" 9 || { echo "[sync_results] lock busy, leaving this for the next sync"; exit 0; }

# We hold the lock, so nothing else is running here: leftovers come from a sync killed by SIGKILL (preemption)
rm -f "$(git rev-parse --git-path index.lock)"
if [ -d "$(git rev-parse --git-path rebase-merge)" ] || [ -d "$(git rev-parse --git-path rebase-apply)" ]; then
    git rebase --abort
fi

git add -A results/ README.md .gitignore
if ! git diff --cached --quiet; then
    git commit -q -m "$MSG" || echo "[sync_results] commit failed"
fi

if timeout 60 git pull -q --rebase origin results && timeout 60 git push -q origin results; then
    echo "[sync_results] pushed $(git rev-parse --short HEAD)"
else
    echo "[sync_results] push failed, commit kept locally; the next sync pushes it"
fi
exit 0
