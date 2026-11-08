#!/bin/zsh
# FRESH campaign (canonical entry point): train -> eval -> gate -> GGUF loop,
# one clean lineage from the base student - no --resume-train /
# --epoch-offset / --eval-resume-source.
#
# Reproducible start-to-finish:
#   code  = origin/main c62f082 - every stage pins the pushed HEAD at its creation
#   data  = the two S3 teacher JSONLs below (pinned objects, produced by
#           teacher_eval/teacher_eval.py runs against the teachers' serving
#           endpoints) + registered SciCode test task 767a2048
#   gate  = eval/scicode_main_pass1 >= 0.05 (teachers score 1.5% glm /
#           10.8% mistral on this metric; 0.15 exceeded every teacher)
#   eval  = thinking-mode, 40000 tokens, temp 0.6 (Qwen thinking recommendation)
#
# Secrets come from .env at the repo root (gitignored). Run with:
#   zsh scripts/launch_campaign_fresh.sh
set -e
cd "$(dirname "$0")/.."
. ./.env
export CML_PROJECT=clearml-finetuning/qwen3.8-coder CLEARML_QUEUE=muller
exec /pscratch/sd/t/tylern/scratch/clearml/.venv/bin/python -u scripts/pipeline.py \
  --from train \
  --upstream-scicode 767a204874474482abeab877586d032a \
  --extra-data "s3://s3-bucket-loadbalancer.das-redpanda.production.svc.spin.nersc.org:5432/dasml/clearml/remote_distills/glm-5.3-flash.jsonl,s3://s3-bucket-loadbalancer.das-redpanda.production.svc.spin.nersc.org:5432/dasml/clearml/remote_distills/mistral-large-4.jsonl" \
  --epochs 1 --max-total-epochs 12 --train-max-len 12288 \
  --eval-gate eval/scicode_main_pass1 --eval-gate-min 0.05 \
  --stop-on-eval-decrease --max-train-retries 2
