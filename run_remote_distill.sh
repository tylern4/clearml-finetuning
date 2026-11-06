#!/usr/bin/env bash
# Run the standalone remote-model distill (scripts/make_remote_distill.py)
# against a local OpenAI-compatible server and upload the JSONL to S3.
#
# Usage:
#   ./run_remote_distill.sh MODEL [LIMIT] [extra make_remote_distill.py args...]
#
#   MODEL   model name as served at the endpoint (required)
#   LIMIT   cap problems; 0 = all (default 0). Try 8 first as a smoke test.
#
# Env overrides:
#   DISTILL_URL     API base          (default http://localhost:9090/v1)
#   DISTILL_TOKEN   bearer token      (default "none" - fine for open servers)
#   DISTILL_INPUT   problem-set jsonl/zip
#                                     (default: the campaign code_data set)
#   DISTILL_S3_URI  full s3 destination (default <S3_BASE>/<model_tag>.jsonl)
#
# The script auto-uploads to S3 at the end (creds come from clearml.conf via
# the StorageManager fallback) and prints the --extra-data line for training.
# Interrupted runs resume from the journal in ~/.cache/remote_distill.

set -euo pipefail

MODEL="${1:?usage: run_remote_distill.sh MODEL [LIMIT] [extra make_remote_distill.py args...]}"
LIMIT="${2:-0}"
if [[ $# -ge 2 ]]; then shift 2; else shift; fi

DISTILL_URL="${DISTILL_URL:-http://localhost:9090/v1}"
DISTILL_TOKEN="${DISTILL_TOKEN:-none}"
DISTILL_INPUT="${DISTILL_INPUT:-s3://s3-bucket-loadbalancer.das-redpanda.production.svc.spin.nersc.org:5432/dasml/clearml/clearml-finetuning/qwen3.8-coder-v2/finetune-code_data.b426ba99ad90419a8243282883ce2457/artifacts/code-data/code.zip}"
S3_BASE="s3://s3-bucket-loadbalancer.das-redpanda.production.svc.spin.nersc.org:5432/dasml/clearml/remote_distills"
TAG="$(echo "$MODEL" | tr '/ ' '__')"
DISTILL_S3_URI="${DISTILL_S3_URI:-$S3_BASE/$TAG.jsonl}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=""
for c in "$HERE/.venv/bin/python" "$HERE/../.venv/bin/python"; do
    if [[ -x "$c" ]]; then PY="$c"; break; fi
done
PY="${PY:-$(command -v python3)}"

echo "== model:   $MODEL"
echo "== url:     $DISTILL_URL"
echo "== input:   $DISTILL_INPUT"
echo "== s3 out:  $DISTILL_S3_URI  (limit=$LIMIT)"

exec "$PY" -u "$HERE/scripts/make_remote_distill.py" \
    --url "$DISTILL_URL" \
    --token "$DISTILL_TOKEN" \
    --model "$MODEL" \
    --input "$DISTILL_INPUT" \
    --s3-uri "$DISTILL_S3_URI" \
    --limit "$LIMIT" \
    "$@"
