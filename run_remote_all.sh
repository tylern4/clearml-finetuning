#!/usr/bin/env bash
# Generate BOTH new-teacher datasets with the local model:
#   1. re-answers of the campaign problem set   (run_remote_distill.sh)
#   2. freshly invented agent tasks + trajectories (run_remote_tasks.sh)
#
# Waits for the endpoint to come up (e.g. until the reverse tunnel is
# established), then runs each generator TWICE - the second pass reuses the
# resume journals and only regenerates problems that returned empty content
# (empties are not journaled, so they get retried for free).
#
# Usage:  ./run_remote_all.sh [MODEL]      (default glm-5.3-flash)
#
# All knobs (URL/token/max-tokens) come from the env just like the two
# individual wrappers; DISTILL_MAX_TOKENS/TASKS_TRAJ_MAX_TOKENS etc. pass
# straight through.

set -uo pipefail

MODEL="${1:-glm-5.3-flash}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$HERE/.env" ]]; then set -a; source "$HERE/.env"; set +a; fi
URL="${DISTILL_URL:-${TASKS_URL:-http://localhost:8989/v1}}"

echo "[wait] polling $URL/models every 15s for the model server..."
while true; do
    resp="$(curl -s --max-time 5 -H "Authorization: Bearer ${REMOTE_API_TOKEN:-none}" \
             "$URL/models" || true)"
    if [[ "$resp" == *'"id"'* ]]; then break; fi
    sleep 15
done
echo "[wait] endpoint is up at $(date)"

for pass in 1 2; do
    echo "[distill pass $pass] start $(date)"
    "$HERE/run_remote_distill.sh" "$MODEL" \
        || echo "[distill pass $pass] FAILED (continuing)"
done
for pass in 1 2; do
    echo "[tasks pass $pass] start $(date)"
    "$HERE/run_remote_tasks.sh" "$MODEL" \
        || echo "[tasks pass $pass] FAILED (continuing)"
done
echo "[done] all passes finished $(date)"
