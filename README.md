# clearml-finetuning

Distill a large open coder model into a coding-agent fine-tune, tracked end-to-end
with a self-hosted [ClearML](https://clear.ml) instance.

The pipeline runs as independent ClearML tasks (one per stage) on a GPU queue.
Each task clones this repo at a pinned commit (the repo is the source of truth),
hands state to the next stage through ClearML artifacts (backed by S3), and
reports scalars/plots/debug samples natively.

## Pipeline

```
distill  ->  train  ->  eval
```

1. **distill** — a large teacher (Qwen3.8-27B, vLLM, TP=4)
   generates synthetic coding-agent tasks *and* full tool-use trajectories under
   a fixed agent scaffold (`run_cmd` / `read_file` / `edit_file` / `finish`
   actions, JSONL format). Output: `train.jsonl` + `val.jsonl` artifacts.
2. **train** — the student (Qwen3.5-9B) is fine-tuned with LoRA
   (peft) on the distilled trajectories, DDP across 4 GPUs, assistant-only loss
   masking. Output: adapter checkpoint artifact.
3. **eval** — merged student is served with vLLM: HumanEval pass@1 (executed)
   plus trajectory-format adherence on held-out agent tasks. Output: metrics +
   sample generations.

## Layout

```
clearml_finetuning/      package: clearml shim, prompts, data helpers
scripts/
  distill_data.py        stage 1: teacher generation
  train_student.py       stage 2: LoRA SFT
  eval_agent.py          stage 3: evaluation
  clearml_stage.py       per-stage task runner (Task.init, artifact handoff)
  pipeline.py            controller: creates/enqueues/polls stage tasks
configs/                 default configs
```

## Running

Requires a ClearML agent on a 4-GPU queue and a ClearML config with S3 access
for artifact storage.

```bash
# controller runs the whole chain (requires the repo to be pushed; the
# agent clones the pinned commit)
python scripts/pipeline.py --all

# or run a single stage (registers + enqueues the task)
python scripts/pipeline.py --only distill
```

Environment knobs (all optional):

| var | meaning | default |
|---|---|---|
| `CLEARML_QUEUE` | agent queue name | `muller` |
| `WF_OUTPUT_URI` | artifact/sample destination (e.g. `s3://bucket/prefix`) | clearml files server |
| `NPROC` | torchrun process count for training | `4` |
| `CML_TEACHER` | teacher model id | `Qwen/Qwen3.8-27B` |
| `CML_STUDENT` | student model id | `Qwen/Qwen3.5-9B` |

## Notes

- Distilled trajectories are *simulated* by the teacher (no real tool
  execution during generation). Executing real tools in a sandboxed loop
  (true agentic distillation) is the obvious v2.
- Models default to Qwen2.5-Coder family; the code is model-agnostic apart
  from the chat template's assistant-mask support.
