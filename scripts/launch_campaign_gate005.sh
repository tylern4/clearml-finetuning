#!/bin/zsh
# HISTORICAL (lineage archived 2026-11-08): the resumed-chunk experiment -
# adopted the in-flight epoch-4 train chunk (3d7a4932) via --resume-train +
# --epoch-offset 3 + --eval-resume-source, gate lowered 0.15 -> 0.05.
# ABANDONED: the epoch-4 eval scored 0/164 HumanEval and 0/291 SciCode with
# healthy train loss and finished generations - the resume lineage silently
# broke the model. That failure is why launch_campaign_fresh.sh trains one
# clean lineage from the base student. Kept for provenance of the archived
# ClearML tasks. Run with:  zsh scripts/launch_campaign_gate005.sh
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
  --stop-on-eval-decrease --max-train-retries 2 \
  --resume-train 3d7a493291534b03a0d950c1bc3ebbb2 \
  --epoch-offset 3 --eval-resume-source
