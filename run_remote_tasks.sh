#!/usr/bin/env bash
# Invent NEW agent tasks + trajectories with a remote OpenAI-compatible model
# (scripts/make_remote_tasks.py) and upload the training JSONL to S3.
#
# Usage:
#   ./run_remote_tasks.sh MODEL [N_TASKS] [extra make_remote_tasks.py args...]
#
#   MODEL    model name as served at the endpoint (required)
#   N_TASKS  task inventions to attempt (default 1200; try 12 as a smoke test)
#
# Env overrides:
#   TASKS_URL      API base        (default http://localhost:8989/v1)
#   TASKS_TOKEN    bearer token    (default: $REMOTE_API_TOKEN from .env)
#   TASKS_S3_URI   full s3 destination for the messages-format JSONL
#                                  (default <S3_BASE>/<model_tag>_tasks.jsonl;
#                                  the raw-transcript archive is uploaded
#                                  next to it as *.transcripts.jsonl)
#   TASKS_GEN_MAX_TOKENS    task-invention budget  (default 800)
#   TASKS_TRAJ_MAX_TOKENS   trajectory budget      (default 12000 - thinking
#                                  models spend reasoning + answer out of ONE
#                                  budget; only the final transcript is kept)
#   TASKS_TIMEOUT   per-request seconds (default 900)
#   TASKS_RETRY_KWARGS  JSON for a rescue pass over empty generations
#                                  (default empty = off; empties retry free on
#                                  rerun since they are not journaled)
#
# The script auto-uploads to S3 at the end (creds come from clearml.conf via
# the StorageManager fallback) and prints the --extra-data line for training.
# Interrupted runs resume from journals in ~/.cache/remote_distill.

set -euo pipefail

MODEL="${1:?usage: run_remote_tasks.sh MODEL [N_TASKS] [extra args...]}"
N_TASKS="${2:-1200}"
if [[ $# -ge 2 ]]; then shift 2; else shift; fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# pick up TASKS_* / REMOTE_API_TOKEN overrides from .env (gitignored) --
# MUST happen before the TASKS_* defaults below are evaluated
if [[ -f "$HERE/.env" ]]; then set -a; source "$HERE/.env"; set +a; fi

TASKS_URL="${TASKS_URL:-http://localhost:8989/v1}"
TASKS_TOKEN="${TASKS_TOKEN:-${REMOTE_API_TOKEN:-none}}"
S3_BASE="s3://s3-bucket-loadbalancer.das-redpanda.production.svc.spin.nersc.org:5432/dasml/clearml/remote_distills"
TAG="$(echo "$MODEL" | tr '/ ' '__')"
TASKS_S3_URI="${TASKS_S3_URI:-$S3_BASE/${TAG}_tasks.jsonl}"

PY=""
for c in "$HERE/.venv/bin/python" "$HERE/../.venv/bin/python"; do
    if [[ -x "$c" ]]; then PY="$c"; break; fi
done
PY="${PY:-$(command -v python3)}"

echo "== model:    $MODEL"
echo "== url:      $TASKS_URL"
echo "== n-tasks:  $N_TASKS"
echo "== s3 out:   $TASKS_S3_URI"

exec "$PY" -u "$HERE/scripts/make_remote_tasks.py" \
    --url "$TASKS_URL" \
    --token "$TASKS_TOKEN" \
    --model "$MODEL" \
    --s3-uri "$TASKS_S3_URI" \
    --n-tasks "$N_TASKS" \
    --task-max-tokens "${TASKS_GEN_MAX_TOKENS:-800}" \
    --traj-max-tokens "${TASKS_TRAJ_MAX_TOKENS:-12000}" \
    --timeout "${TASKS_TIMEOUT:-900}" \
    --retry-kwargs "${TASKS_RETRY_KWARGS:-}" \
    "$@"
