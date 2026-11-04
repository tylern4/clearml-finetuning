"""Stage: distill teacher solutions for the real code-domain problems.

Takes the code_data stage's code_train.jsonl, strips each row's reference
assistant answer, and has the teacher model generate its own assistant turn
for the same prompt (system + preceding context). Output rows use the same
{"messages": [...]} SFT format and are written to code_distilled.jsonl,
uploaded as the `code-distilled-data` artifact. The train stage mixes these
teacher-solved rows in alongside the synthetic distill / scicode / raw code
data.

Uses vLLM (tensor-parallel) like the distill stage. Generation runs in
batches with flushed progress prints and faulthandler enabled so that a
native engine crash pinpoints the failing batch instead of dying silently.
"""

import argparse
import faulthandler  # dumps native stack traces on SIGSEGV/SIGABRT to stderr
import os
import random
import sys
import traceback

faulthandler.enable()
sys.stdout.reconfigure(line_buffering=True)  # never lose a progress print

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
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=2400)
    ap.add_argument("--batch-size", type=int, default=128,
                    help="problems per llm.generate call (progress granularity)")
    ap.add_argument("--out-dir", default="code_distilled")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    run = cml.get_run_logger("code_distill", vars(args))

    def logp(msg):
        print(msg, flush=True)

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
    logp(f"solving {len(rows)} code problems with teacher {args.teacher}")

    # ---- prompt prep BEFORE loading the engine (tokenizer only, CPU) ----------
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.teacher)

    def chat(messages):
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    # prompt = every message except the reference final assistant answer,
    # trimmed to fit the model context (SWE-chat rows carry up to 8 x 4k-char
    # messages and otherwise blow past max_model_len, killing the engine)
    budget = args.max_model_len - args.max_tokens - 64

    def ntok(messages):
        return len(tok(chat(messages), add_special_tokens=False)["input_ids"])

    def fit_messages(messages):
        """Trim until the templated prompt fits the token budget; None = unsalvageable."""
        messages = data.normalize_messages([dict(m) for m in messages])
        while len(messages) > 2 and ntok(messages) > budget:
            # drop the oldest non-system turn
            del messages[1 if messages[0]["role"] == "system" else 0]
        if ntok(messages) <= budget:
            return messages
        # only [system?, user] left and still too long: truncate the user text
        user_ids = tok(messages[-1]["content"], add_special_tokens=False)["input_ids"]
        room = budget - (ntok(messages) - len(user_ids))
        if room <= 0:
            return None
        messages[-1]["content"] = tok.decode(user_ids[:room]) + "\n... [truncated]"
        return messages if ntok(messages) <= budget else None

    prompts, kept_rows, fit_msgs = [], [], []
    n_trim, n_drop = 0, 0
    for r in rows:
        msgs = r["messages"]
        if msgs and msgs[-1]["role"] == "assistant":
            msgs = msgs[:-1]
        if not any(m["role"] == "user" for m in msgs):
            continue
        try:
            fitted = fit_messages(msgs)
        except Exception as e:
            # one malformed row must not kill the stage
            logp(f"row {len(prompts)} unusable ({type(e).__name__}: {e}); skipped")
            n_drop += 1
            continue
        if not fitted:
            n_drop += 1
            continue
        if len(fitted) != len(msgs) or fitted[-1]["content"] != msgs[-1]["content"]:
            n_trim += 1
        prompts.append(chat(fitted))
        fit_msgs.append(fitted)
        kept_rows.append(r)
    logp(f"prompts: {len(prompts)} usable, {n_trim} context-trimmed, "
         f"{n_drop} dropped (budget {budget} tokens)")
    lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    logp(f"prompt tokens: min {min(lens)}, median {sorted(lens)[len(lens)//2]}, "
         f"max {max(lens)} (limit {budget})")

    # ---- engine ---------------------------------------------------------------
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

    sp = SamplingParams(temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_tokens)

    bs = max(1, args.batch_size)
    n_batches = (len(prompts) + bs - 1) // bs
    outs = []
    for bi in range(n_batches):
        lo, hi = bi * bs, min((bi + 1) * bs, len(prompts))
        logp(f"[batch {bi + 1}/{n_batches}] generating rows {lo}..{hi - 1} "
             f"({hi - lo} prompts, max {max(lens[lo:hi])} tokens)")
        try:
            outs.extend(llm.generate(prompts[lo:hi], sp))
        except Exception:
            logp(f"[batch {bi + 1}/{n_batches}] GENERATE FAILED on rows {lo}..{hi - 1}; "
                 f"traceback follows")
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            raise
        step = len(outs)
        if run:
            run.log({"code_distill/batches_done": bi + 1,
                     "code_distill/solved_so_far": sum(
                         1 for o in outs if o.outputs[0].text.strip())}, step=step)
        logp(f"[batch {bi + 1}/{n_batches}] done ({len(outs)}/{len(prompts)} total)")
        # write partial results so progress survives a later crash
        os.makedirs(args.out_dir, exist_ok=True)
        data.write_jsonl(os.path.join(args.out_dir, "code_distilled.jsonl"),
                         build_rows(kept_rows, fit_msgs, outs))

    out_rows = build_rows(kept_rows, fit_msgs, outs)
    n_empty = len(kept_rows) - len(out_rows)
    logp(f"teacher solved {len(out_rows)}/{len(kept_rows)} problems "
         f"({n_empty} empty generations dropped)")

    out_path = os.path.join(args.out_dir, "code_distilled.jsonl")
    data.write_jsonl(out_path, out_rows)
    logp(f"wrote {len(out_rows)} rows to {out_path}")

    if run:
        run.log({"code_distill/problems": len(kept_rows),
                 "code_distill/trimmed": n_trim,
                 "code_distill/dropped": n_drop,
                 "code_distill/solved": len(out_rows),
                 "code_distill/empty": n_empty}, step=len(out_rows))
        from pathlib import Path
        run.task.upload_artifact("code-distilled-data", Path(args.out_dir),
                                 wait_on_upload=True)
        run.finish()


def build_rows(kept_rows, fit_msgs, outs):
    """SFT rows from the teacher generations (strip empty ones).

    zip truncates to the shortest list, so partial `outs` after a mid-run
    crash still yield the rows solved so far.
    """
    rows = []
    for r, msgs, out in zip(kept_rows, fit_msgs, outs):
        text = out.outputs[0].text.strip()
        if not text:
            continue
        rows.append({"messages": msgs + [{"role": "assistant", "content": text}],
                     "meta": {**(r.get("meta") or {}), "source": (
                         (r.get("meta") or {}).get("source", "?") + "-teacher")}})
    return rows


if __name__ == "__main__":
    main()
