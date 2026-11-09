#!/usr/bin/env bash
# Distill one or more OCR problem-set batches through a single remote teacher,
# each batch followed by a low-effort rescue pass over its empty generations.
#
# Usage:
#   DISTILL_URL=http://localhost:8989/v1 \
#     scripts/run_ocr_batches.sh glm-5.3-flash \
#     s3://.../ocr_problem_set.jsonl s3://.../ocr_problem_set_b2.jsonl
#
# For an input named ocr_problem_set<TAG>.jsonl the rows land at
# <S3_BASE>/ocr-<model><TAG>.jsonl (batch 1 keeps its existing URI), and the
# local working copy at remote_distill_ocr-<model><TAG>.jsonl.
#
# Env:
#   DISTILL_URL            API base of the teacher (required)
#   DISTILL_MAX_TOKENS     completion budget        (default 12288)
#   DISTILL_TIMEOUT        per-request seconds      (default 1200)
#   DISTILL_TRAIN_MAX_LEN  student window, report   (default 16384)
#   DISTILL_CONCURRENCY    parallel requests        (default 10)
#   DISTILL_TEMPERATURE    sampling temperature     (default 0.3)
#   RESCUE_KWARGS          JSON merged into the request body of the rescue
#                          pass (default {"reasoning_effort":"low"}; empty
#                          string skips the rescue pass)
#
# Every step is idempotent via the resume journal: pass 1 skips journaled
# successes, and the rescue rerun re-samples only the empty generations
# (a fresh default-effort sample first, then the rescue kwargs). Rescued
# rows keep a reasoning trace - just a shorter one - so the R1-style row
# format is preserved.

set -euo pipefail

MODEL="$1"; shift
[[ $# -ge 1 ]] || { echo "usage: $0 MODEL SET [SET ...]" >&2; exit 2; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S3_BASE="s3://s3-bucket-loadbalancer.das-redpanda.production.svc.spin.nersc.org:5432/dasml/clearml/remote_distills"

if [[ -z "${RESCUE_KWARGS+x}" ]]; then
  RESCUE_KWARGS='{"reasoning_effort":"low"}'
fi

MODEL_TAG="$(echo "$MODEL" | tr '/ ' '__')"
outs=()
for SET in "$@"; do
  base="$(basename "$SET")"
  tag="${base#ocr_problem_set}"; tag="${tag%.jsonl}"
  out="$S3_BASE/ocr-${MODEL_TAG}${tag}.jsonl"
  local_out="remote_distill_ocr-${MODEL_TAG}${tag}.jsonl"
  outs+=("$out")

  echo "== [$(date +%H:%M:%S)] batch $base -> $out (pass 1)"
  DISTILL_URL="${DISTILL_URL:?set DISTILL_URL}" \
  DISTILL_INPUT="$SET" DISTILL_S3_URI="$out" \
  DISTILL_MAX_TOKENS="${DISTILL_MAX_TOKENS:-12288}" \
  DISTILL_TIMEOUT="${DISTILL_TIMEOUT:-1200}" \
  DISTILL_TRAIN_MAX_LEN="${DISTILL_TRAIN_MAX_LEN:-16384}" \
    "$REPO/run_remote_distill.sh" "$MODEL" 0 \
    --concurrency "${DISTILL_CONCURRENCY:-10}" \
    --temperature "${DISTILL_TEMPERATURE:-0.3}" \
    --out "$local_out"

  if [[ -n "$RESCUE_KWARGS" ]]; then
    echo "== [$(date +%H:%M:%S)] batch $base -> $out (rescue: $RESCUE_KWARGS)"
    DISTILL_URL="${DISTILL_URL:?set DISTILL_URL}" \
    DISTILL_INPUT="$SET" DISTILL_S3_URI="$out" \
    DISTILL_MAX_TOKENS="${DISTILL_MAX_TOKENS:-12288}" \
    DISTILL_TIMEOUT="${DISTILL_TIMEOUT:-1200}" \
    DISTILL_TRAIN_MAX_LEN="${DISTILL_TRAIN_MAX_LEN:-16384}" \
    DISTILL_RETRY_KWARGS="$RESCUE_KWARGS" \
      "$REPO/run_remote_distill.sh" "$MODEL" 0 \
      --concurrency "${DISTILL_CONCURRENCY:-10}" \
      --temperature "${DISTILL_TEMPERATURE:-0.3}" \
      --out "$local_out"
  fi
done

echo "== [$(date +%H:%M:%S)] $MODEL done; extra-data URIs:"
( IFS=,; echo "${outs[*]}" )
