# clearml-finetuning

Distill a large open coder model into a coding-agent fine-tune, tracked end-to-end
with a self-hosted [ClearML](https://clear.ml) instance.

The pipeline runs as independent ClearML tasks (one per stage) on a GPU queue.
Each task clones this repo at a pinned commit (the repo is the source of truth),
hands state to the next stage through ClearML artifacts (backed by S3), and
reports scalars/plots/debug samples natively.

## Pipeline

```
distill  ->  train  ->  eval  ->  convert_gguf
```

1. **distill** — a large teacher (default `Qwen/Qwen3.8-27B`, vLLM, TP=4)
   generates synthetic coding-agent tasks *and* full tool-use trajectories under
   a fixed agent scaffold (`run_cmd` / `read_file` / `edit_file` / `finish`
   actions, JSONL format). Output: `train.jsonl` + `val.jsonl` artifacts.
   Trajectories distilled from other teachers (e.g. served-endpoint runs) can
   be injected via `--extra-data s3://…` without rerunning this stage.
2. **train** — the student (default `Qwen/Qwen3.5-9B`) is fine-tuned with LoRA
   (peft) on the distilled trajectories, DDP across 4 GPUs, assistant-only loss
   masking. Output: adapter checkpoint artifact.
3. **eval** — merged student is served with vLLM: HumanEval pass@1 (executed)
   plus SciCode sub-step pass@1 and main resolved@1, thinking-mode with a
   distill-matched token budget. Output: metrics + sample generations.
4. **convert_gguf** — merge the LoRA adapter into the base student, convert
   to GGUF (pinned llama.cpp release) and quantize (Q4_K_M + Q8_0) so the
   fine-tuned model runs on a laptop with llama.cpp/Ollama. GGUF files are
   uploaded as artifacts.

SciCode test data is provided to eval by a `scicode_data` stage (registered
upstream via `--upstream-scicode`). The controller can gate the epoch loop on
any reported scalar (e.g. `--eval-gate eval/scicode_main_pass1
--eval-gate-min 0.05`), stopping when the metric drops.

The active campaign (`scripts/launch_campaign_fresh.sh`) self-distills
`Qwen/Qwen3.8-27B` (teacher = student): the trajectory JSONLs distilled from
the served teachers (glm-5.3-flash, mistral-large-4) enter through
`--extra-data`, and the gate is `eval/scicode_main_pass1 >= 0.05` — both
teachers score 1.5% / 10.8% on that metric, so the gate sits below them.

## Layout

```
clearml_finetuning/      package: clearml shim, prompts, data helpers
scripts/
  distill_data.py        stage: teacher generation
  scicode_data.py        stage: registers the SciCode test set for eval
  code_data.py           stage: real-code task data
  code_distill.py        stage: distillation against real code tasks
  train_student.py       stage: LoRA SFT
  eval_agent.py          stage: evaluation (HumanEval + SciCode)
  convert_gguf.py        stage: merge + GGUF quantization (side task)
  clearml_stage.py       per-stage task runner (Task.init, artifact handoff)
  pipeline.py            controller: creates/enqueues/polls stage tasks
  launch_campaign_*.sh   committed campaign launchers (fresh = canonical)
configs/                 default configs
```

## Running

Requires a ClearML agent on a 4-GPU queue and a ClearML config with S3 access
for artifact storage.

```bash
# full campaign as committed (train -> eval -> gate -> gguf loop, fresh
# lineage; sources .env for credentials):
zsh scripts/launch_campaign_fresh.sh

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

## Running on a laptop (llama.cpp)

The final stage publishes quantized GGUF files as artifacts
(`<student>-agent-Q4_K_M.gguf` ≈ 5.5 GB, `Q8_0` ≈ 9.5 GB). Download from the
task's Artifacts tab, then:

```bash
# interactive chat (the GGUF embeds the model's chat template)
llama-cli -m qwen3.5-9b-agent-Q4_K_M.gguf -c 8192

# one-shot completion, no prompt echo
llama-cli -m qwen3.5-9b-agent-Q4_K_M.gguf -c 8192 \
    -p "Fix the failing test in tests/test_retry.py" --no-display-prompt
```

The model was fine-tuned against a fixed agent scaffold (JSONL tool actions:
`run_cmd` / `read_file` / `edit_file` / `finish`); to get scaffold-formatted
responses, pass the system prompt from `clearml_finetuning/prompts.py`
(`AGENT_SYSTEM_PROMPT`) as the system message. Easiest via the built-in
OpenAI-compatible server:

```bash
llama-server -m qwen3.5-9b-agent-Q4_K_M.gguf -c 8192

curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system","content":"<AGENT_SYSTEM_PROMPT>"},{"role":"user","content":"<task>"}]}'
```

Notes: add `-ngl 99` to offload all layers if the laptop has a CUDA/Metal
GPU; CPU-only inference with Q4_K_M runs at usable speeds for a 9B model.

## Notes

- Distilled trajectories are *simulated* by the teacher (no real tool
  execution during generation). Executing real tools in a sandboxed loop
  (true agentic distillation) is the obvious v2.
- Models default to the Qwen3 family (`Qwen/Qwen3.8-27B` teacher,
  `Qwen/Qwen3.5-9B` student); the code is model-agnostic apart from the chat
  template's assistant-mask support.
