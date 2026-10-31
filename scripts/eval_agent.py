"""Stage: evaluate the fine-tuned student.

1. HumanEval pass@1 (greedy, executed in subprocesses with a timeout).
2. Trajectory-format adherence on held-out distilled agent tasks: the student
   must emit valid JSONL agent actions under the scaffold.
Reports metrics to ClearML and uploads sample generations.
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gzip
import urllib.request

from clearml_finetuning import cml, data, prompts  # noqa: E402

_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>.*$", re.DOTALL | re.IGNORECASE)


def strip_think(text):
    """Remove reasoning traces from model output (defensive)."""
    if not isinstance(text, str):
        return text
    text = _THINK_RE.sub("", text)
    return _THINK_OPEN_RE.sub("", text).strip()


def chat(tok, messages):
    """Chat-template with thinking DISABLED.

    Qwen3.5's default template puts the model in thinking mode (the generation
    prompt ends with an open think block). With a 512-token budget that burns
    everything on reasoning and leaves no code, so eval must disable it
    (same as the distill stage).
    """
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def load_humaneval(attempts=3):
    """Fetch HumanEval directly (avoids the datasets/hub version matrix)."""
    urls = [
        "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
        "https://raw.githubusercontent.com/openai/human-eval/main/data/HumanEval.jsonl.gz",
    ]
    last = None
    for url in urls:
        for i in range(attempts):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    blob = gzip.decompress(r.read())
                return [json.loads(l) for l in blob.decode().splitlines() if l.strip()]
            except Exception as e:
                last = e
                print(f"HumanEval download failed ({e}); retry {i + 1}/{attempts}")
    raise last


def main() -> None:
    ap = argparse.ArgumentParser(description="evaluate student")
    ap.add_argument("--student", default=os.environ.get("CML_STUDENT",
                    "Qwen/Qwen3.5-9B"))
    ap.add_argument("--adapter-task", type=str, default="", help="upstream train task id")
    ap.add_argument("--adapter-dir", type=str, default="student_adapter")
    ap.add_argument("--data-task", type=str, default="", help="upstream distill task id (for val rows)")
    ap.add_argument("--data-dir", type=str, default="distilled")
    ap.add_argument("--humaneval-n", type=int, default=0, help="0 = all 164 problems")
    ap.add_argument("--adherence-n", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    run = cml.get_run_logger("eval", vars(args), task_type="testing")

    from vllm import LLM
    from vllm.lora.request import LoRARequest

    nproc = int(os.environ.get("NPROC", "4"))
    if args.adapter_task and not os.path.isdir(args.adapter_dir):
        local = cml.download_artifact(args.adapter_task, "student-adapter")
        os.makedirs(os.path.dirname(args.adapter_dir) or ".", exist_ok=True)
        os.rename(local, args.adapter_dir) if not os.path.exists(args.adapter_dir) else None
    use_lora = os.path.isdir(args.adapter_dir) and os.path.exists(
        os.path.join(args.adapter_dir, "adapter_config.json"))

    llm = LLM(
        model=args.student,
        tensor_parallel_size=max(1, min(nproc, 4)),
        max_model_len=8192,
        enable_lora=use_lora,
        max_lora_rank=64 if use_lora else None,
        gpu_memory_utilization=0.90,
    )
    tok = llm.get_tokenizer()
    lora_req = LoRARequest("student", 1, args.adapter_dir) if use_lora else None

    # ---- HumanEval pass@1 (with adapter if present) --------------------------
    problems = load_humaneval()
    problems = problems[:args.humaneval_n] if args.humaneval_n else problems
    prompts_ = [
        chat(tok, [{"role": "user", "content": p["prompt"] +
                    "\n\nComplete the following function. Return only the code."}])
        for p in problems
    ]
    sp = llm.get_default_sampling_params()
    sp.temperature = 0.0
    sp.max_tokens = args.max_tokens
    outs = llm.generate(prompts_, sp, lora_request=lora_req)

    passed = 0
    for p, out in zip(problems, outs):
        code = strip_think(out.outputs[0].text)
        if "```" in code:
            blocks = [b for b in code.split("```") if b.strip()]
            code = blocks[0].split("\n", 1)[-1] if blocks else code
        body = p["prompt"] + "\n" + code
        check = p["test"] + f"\ncheck({p['entry_point']})\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write("import math, os, re, sys, string, itertools\n" + body + "\n" + check)
            path = f.name
        try:
            r = subprocess.run([sys.executable, path], capture_output=True, timeout=15)
            passed += int(r.returncode == 0)
        except subprocess.TimeoutExpired:
            pass
        finally:
            os.unlink(path)
    pass1 = passed / max(1, len(problems))
    print(f"HumanEval pass@1: {passed}/{len(problems)} = {pass1:.3f}")
    if run:
        run.log({"eval/humaneval_passed": passed, "eval/humaneval_n": len(problems),
                 "eval/pass1": pass1}, step=0)

    # ---- trajectory format adherence ----------------------------------------
    if args.data_task and not os.path.isdir(args.data_dir):
        local = cml.download_artifact(args.data_task, "distilled-data")
        os.makedirs(args.data_dir, exist_ok=True)
        import shutil
        for f in os.listdir(local):
            shutil.copy(os.path.join(local, f), args.data_dir)
    val_rows = data.read_jsonl(os.path.join(args.data_dir, "val.jsonl"))
    system_prompt = (val_rows[0]["messages"][0]["content"]
                     if val_rows and "messages" in val_rows[0]
                     else prompts.AGENT_SYSTEM_PROMPT)

    import random

    rng = random.Random(0)
    sample = rng.sample(val_rows, min(args.adherence_n, len(val_rows))) if val_rows else []
    agent_prompts = [
        chat(tok, [{"role": "system", "content": system_prompt},
                   {"role": "user", "content": f"{r['repo_context']}\n\nTask: {r['task']}"}])
        for r in sample
    ]
    sp2 = llm.get_default_sampling_params()
    sp2.temperature = 0.0
    sp2.max_tokens = 1400
    outs2 = llm.generate(agent_prompts, sp2, lora_request=lora_req)

    valid = 0
    for row, out in zip(sample, outs2):
        text = strip_think(out.outputs[0].text)
        if data.validate_transcript(text) is not None:
            valid += 1
        elif run:
            run.report_sample(f"invalid transcript for '{row['title']}':\n{text[:900]}",
                              step=1, label=f"invalid_{row['fingerprint']}")
    frac = valid / max(1, len(sample))
    print(f"trajectory adherence: {valid}/{len(sample)} = {frac:.3f}")
    if run:
        run.log({"eval/agent_tasks": len(sample), "eval/valid_trajectories": valid,
                 "eval/format_adherence": frac}, step=1)

    # ---- a couple of end-to-end agent samples as debug samples ---------------
    demo_prompts = [
        "Fix the bug in utils/date.py where Feb 29 crashes the parser.",
        "Add a unit test for the retry helper in client/retry.py.",
    ]
    for i, dp in enumerate(demo_prompts):
        ptext = chat(tok, [{"role": "system", "content": system_prompt},
                           {"role": "user", "content": dp}])
        out = strip_think(llm.generate([ptext], sp2, lora_request=lora_req)[0].outputs[0].text)
        if run:
            run.report_sample(f"task: {dp}\n\nstudent actions:\n{out[:1200]}",
                              step=2 + i, label=f"demo_{i}")

    if run:
        run.finish()


if __name__ == "__main__":
    main()
