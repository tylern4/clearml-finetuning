"""Stage: distill teacher solutions for the real code-domain problems.

Takes the code_data stage's code_train.jsonl, strips each row's reference
assistant answer, and has the teacher model generate its own assistant turn
for the same prompt (system + preceding context). Output rows use the same
{"messages": [...]} SFT format and are written to code_distilled.jsonl,
uploaded as the `code-distilled-data` artifact. The train stage mixes these
teacher-solved rows in alongside the synthetic distill / scicode / raw code
data.

Uses vLLM (tensor-parallel) like the distill stage.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clearml_finetuning import cml, data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="teacher-solve the code_data problems")
    ap.add_argument("--code-task", default="", help="upstream code_data task id")
    ap.add_argument("--code-dir", default="code", help="local code-data dir if present")
    ap.add_argument("--teacher", default=os.environ.get("CML_TEACHER",
                     "Qwen/Qwen3.8-27B"))
    ap.add_argument("--max-rows", type=int, default=0,
                    help="cap on problems to solve; 0 = all rows in code_train.jsonl")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=2400)
    ap.add_argument("--out-dir", default="code_distilled")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    run = cml.get_run_logger("code_distill", vars(args))

    # ---- load the code_data problems -----------------------------------------
    import shutil
    code_train = os.path.join(args.code_dir, "code_train.jsonl")
    if args.code_task and not os.path.exists(code_train):
        local = cml.download_artifact(args.code_task, "code-data")
        os.makedirs(args.code_dir, exist_ok=True)
        if os.path.isdir(local):
            for f in os.listdir(local):
                shutil.copy(os.path.join(local, f), args.code_dir)
        else:
            shutil.copy(local, code_train)
    if not os.path.exists(code_train):
        raise RuntimeError(f"no code_train.jsonl in {args.code_dir} and no --code-task")
    rows = data.read_jsonl(code_train)
    if args.max_rows > 0:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.max_rows]
    print(f"solving {len(rows)} code problems with teacher {args.teacher}")

    cml.prepare_vllm_nvrtc()  # FP8 teachers JIT via NVRTC; runtime image lacks it
    from vllm import LLM, SamplingParams

    nproc = int(os.environ.get("NPROC", "4"))
    llm = LLM(
        model=args.teacher,
        tensor_parallel_size=max(1, min(nproc, 4)),
        max_model_len=args.max_model_len,
        quantization="awq" if "awq" in args.teacher.lower() else None,
        gpu_memory_utilization=float(os.environ.get("CML_GPU_MEM_UTIL", "0.90")),
    )
    tok = llm.get_tokenizer()

    def chat(messages):
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    # prompt = every message except the reference final assistant answer
    prompts, kept_rows = [], []
    for r in rows:
        msgs = r["messages"]
        if msgs and msgs[-1]["role"] == "assistant":
            msgs = msgs[:-1]
        if not any(m["role"] == "user" for m in msgs):
            continue
        prompts.append(msgs)
        kept_rows.append(r)

    sp = SamplingParams(temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_tokens)
    outs = llm.generate([chat(m) for m in prompts], sp)

    out_rows, n_empty = [], 0
    for r, msgs, out in zip(kept_rows, prompts, outs):
        text = out.outputs[0].text.strip()
        if not text:
            n_empty += 1
            continue
        out_rows.append({"messages": msgs + [{"role": "assistant", "content": text}],
                         "meta": {**(r.get("meta") or {}), "source": (
                             (r.get("meta") or {}).get("source", "?") + "-teacher")}})
    print(f"teacher solved {len(out_rows)}/{len(kept_rows)} problems "
          f"({n_empty} empty generations dropped)")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "code_distilled.jsonl")
    data.write_jsonl(out_path, out_rows)
    print(f"wrote {len(out_rows)} rows to {out_path}")

    if run:
        run.log({"code_distill/problems": len(kept_rows),
                 "code_distill/solved": len(out_rows),
                 "code_distill/empty": n_empty}, step=0)
        from pathlib import Path
        run.task.upload_artifact("code-distilled-data", Path(args.out_dir),
                                 wait_on_upload=True)
        run.finish()


if __name__ == "__main__":
    main()
