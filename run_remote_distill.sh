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
#   DISTILL_URL     API base          (default http://localhost:8989/v1)
#   DISTILL_TOKEN   bearer token      (default: $REMOTE_API_TOKEN from .env)
#   DISTILL_INPUT   problem-set jsonl/zip
#                                     (default: the campaign code_data set)
#   DISTILL_S3_URI  full s3 destination (default <S3_BASE>/<model_tag>.jsonl)
#   DISTILL_MAX_TOKENS  completion budget (default 40000 - thinking models spend
#                      reasoning + answer out of ONE budget; BOTH are kept:
#                      the reasoning is stored think-wrapped ahead of the
#                      answer so the student learns to reason)
#   DISTILL_TIMEOUT     per-request seconds (default 3600 - 40k-token
#                      generations plus queueing need the headroom)
#   DISTILL_RETRY_KWARGS  JSON merged into the request body for a rescue pass
#                      over empty generations. Default: off - rows without a
#                      final answer are dropped and retried free on rerun.
#                      NOTE: rescue kwargs that bound thinking (e.g. thinking
#                      false) would store rows WITHOUT a reasoning trace
#   DISTILL_TRAIN_MAX_LEN  student --max-len, for the row-size report
#                      (default 49152 - raised with the budget so recovered
#                      long-thinking rows still fit the training window)
#
# The script auto-uploads to S3 at the end (creds come from clearml.conf via
# the StorageManager fallback) and prints the --extra-data line for training.
# Interrupted runs resume from the journal in ~/.cache/remote_distill.

set -euo pipefail

MODEL="${1:?usage: run_remote_distill.sh MODEL [LIMIT] [extra make_remote_distill.py args...]}"
LIMIT="${2:-0}"
if [[ $# -ge 2 ]]; then shift 2; else shift; fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# pick up DISTILL_* / REMOTE_API_TOKEN overrides from .env (gitignored) --
# MUST happen before the DISTILL_* defaults below are evaluated
if [[ -f "$HERE/.env" ]]; then set -a; source "$HERE/.env"; set +a; fi

DISTILL_URL="${DISTILL_URL:-http://localhost:8989/v1}"
DISTILL_TOKEN="${DISTILL_TOKEN:-${REMOTE_API_TOKEN:-none}}"
DISTILL_INPUT="${DISTILL_INPUT:-s3://s3-bucket-loadbalancer.das-redpanda.production.svc.spin.nersc.org:5432/dasml/clearml/clearml-finetuning/qwen3.8-coder-v2/finetune-code_data.b426ba99ad90419a8243282883ce2457/artifacts/code-data/code.zip}"
S3_BASE="s3://s3-bucket-loadbalancer.das-redpanda.production.svc.spin.nersc.org:5432/dasml/clearml/remote_distills"
TAG="$(echo "$MODEL" | tr '/ ' '__')"
DISTILL_S3_URI="${DISTILL_S3_URI:-$S3_BASE/$TAG.jsonl}"

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
    --max-tokens "${DISTILL_MAX_TOKENS:-40000}" \
    --timeout "${DISTILL_TIMEOUT:-3600}" \
    --retry-kwargs "${DISTILL_RETRY_KWARGS-''}" \
    --train-max-len "${DISTILL_TRAIN_MAX_LEN:-49152}" \
    "$@"
