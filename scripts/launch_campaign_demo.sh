#!/bin/zsh
# HISTORICAL (lineage archived 2026-11-08): the original demo campaign at gate
# 0.15. Its evals ran the trajectory-adherence metric (removed in c62f082) and
# the gate value exceeded every teacher (glm 1.5% / mistral 10.8% on
# eval/scicode_main_pass1). Kept for provenance of the archived ClearML tasks.
# Run with:  zsh scripts/launch_campaign_demo.sh
set -e
cd "$(dirname "$0")/.."
. ./.env
export CML_PROJECT=clearml-finetuning/qwen3.8-coder CLEARML_QUEUE=muller
exec /pscratch/sd/t/tylern/scratch/clearml/.venv/bin/python -u scripts/pipeline.py \
  --from train \
  --upstream-scicode 767a204874474482abeab877586d032a \
  --extra-data "s3://s3-bucket-loadbalancer.das-redpanda.production.svc.spin.nersc.org:5432/dasml/clearml/remote_distills/glm-5.3-flash.jsonl,s3://s3-bucket-loadbalancer.das-redpanda.production.svc.spin.nersc.org:5432/dasml/clearml/remote_distills/mistral-large-4.jsonl" \
  --epochs 1 --max-total-epochs 12 --train-max-len 12288 \
  --eval-gate eval/scicode_main_pass1 --eval-gate-min 0.15 \
  --stop-on-eval-decrease --max-train-retries 2
