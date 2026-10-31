"""Stage: distill coding-agent training data from a large teacher model.

Uses vLLM (tensor-parallel) to (1) invent diverse, realistic coding-agent
tasks and (2) produce full tool-use trajectories for each task under the
agent scaffold. Outputs validated, deduplicated JSONL train/val artifacts.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clearml_finetuning import cml, data, prompts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="distill agent trajectories")
    ap.add_argument("--teacher", default=os.environ.get("CML_TEACHER",
                    "Qwen/Qwen3.8-27B"))
    ap.add_argument("--n-tasks", type=int, default=1200, help="task inventions to attempt")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--out-dir", default="distilled")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    run = cml.get_run_logger("distill", vars(args))
    from vllm import LLM, SamplingParams

    nproc = int(os.environ.get("NPROC", "4"))
    llm = LLM(
        model=args.teacher,
        tensor_parallel_size=max(1, min(nproc, 4)),
        max_model_len=args.max_model_len,
        quantization="awq" if "awq" in args.teacher.lower() else None,
        gpu_memory_utilization=0.90,
    )
    tok = llm.get_tokenizer()

    def chat(messages):
        """Template a chat; disable thinking when the template supports it."""
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    rng = random.Random(args.seed)

    # ---- pass 1: invent tasks ------------------------------------------------
    task_prompts = [prompts.sample_task_prompt(rng) for _ in range(args.n_tasks)]
    sp_tasks = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=400)
    outs = llm.generate([chat([{"role": "user", "content": p}]) for p in task_prompts], sp_tasks)

    tasks, seen, rejects = [], set(), []
    for out in outs:
        text = out.outputs[0].text
        obj = data.extract_json_object(text)
        cleaned = data.validate_task_obj(obj) if obj else None
        if not cleaned:
            if len(rejects) < 3:
                rejects.append(text[:800])
            continue
        fp = data.task_fingerprint(cleaned)
        if fp in seen:
            continue
        seen.add(fp)
        cleaned["fingerprint"] = fp
        tasks.append(cleaned)
    if run and rejects:
        for i, r in enumerate(rejects):
            run.report_sample(f"rejected task-invention output:\n{r}", step=0,
                              label=f"reject_task_{i}")

    print(f"kept {len(tasks)}/{args.n_tasks} invented tasks after validation+dedup")
    if run:
        run.log({"distill/tasks_invented": args.n_tasks, "distill/tasks_kept": len(tasks)}, step=0)

    # ---- pass 2: generate trajectories --------------------------------------
    sp_traj = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=2400)
    traj_prompts = [
        [{"role": "system", "content": prompts.AGENT_SYSTEM_PROMPT},
         {"role": "user", "content": prompts.TRAJECTORY_INSTRUCTION.format(
             repo_context=t["repo_context"], task=t["task"])}]
        for t in tasks
    ]
    outs = llm.generate([chat(m) for m in traj_prompts], sp_traj)

    rows, traj_rejects = [], []
    for task, out in zip(tasks, outs):
        text = out.outputs[0].text
        transcript = data.validate_transcript(text)
        if not transcript:
            if len(traj_rejects) < 3:
                traj_rejects.append(text[:900])
            continue
        rows.append({**task, "transcript": transcript})
    if run and traj_rejects:
        for i, r in enumerate(traj_rejects):
            run.report_sample(f"rejected trajectory for '{tasks[i]['title']}':\n{r}",
                              step=1, label=f"reject_traj_{i}")

    print(f"kept {len(rows)}/{len(tasks)} trajectories after validation")
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_fraction))
    val, train = rows[:n_val], rows[n_val:]

    data.write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    data.write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val)
    print(f"wrote {len(train)} train / {len(val)} val rows to {args.out_dir}/")

    if run:
        run.log({
            "distill/trajectories_kept": len(rows),
            "distill/train_rows": len(train),
            "distill/val_rows": len(val),
        }, step=1)
        from pathlib import Path
        run.task.upload_artifact("distilled-data", Path(args.out_dir), wait_on_upload=True)
    run.finish() if run else None


if __name__ == "__main__":
    main()
