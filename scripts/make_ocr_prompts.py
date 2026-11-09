"""Build a competitive-programming problem set from NVIDIA OpenCodeReasoning.

Emits the same {"messages": [...], "meta": {...}} JSONL the code_data stage
produces, so scripts/make_remote_distill.py can distill it from any remote
model: each row is [system, user, reference-assistant]; the reference answer
(OCR's own solution) is stripped/replaced by the remote model at distill time.

Problem selection favors the student's 12288-token training window: only
easy/medium difficulty (thinking traces stay short), problem statements
<= --max-input-chars, and a per-source cap so codeforces cannot dominate.

Examples
--------
  python scripts/make_ocr_prompts.py --n 1250 --out ocr_problem_set.jsonl
"""

import argparse
import json
import random
import sys

SYS_PROMPT = """You are an expert competitive programmer. Write a complete Python 3 program that solves the given problem: read input from stdin exactly as the problem specifies and print the required output.

Rules:
- Output only the Python code - no commentary, no markdown fences.
- Read from stdin and print the answer; no interactive prompts, no file I/O.
- The solution must meet the time and memory limits implied by the constraints."""

# keep: explicit easy-side labels + small numeric bins (codeforces-style)
KEEP_LABELS = {"EASY", "MEDIUM", "interview"}
MAX_NUMERIC = 8.0
# drop the ambiguous middle/hard labels: MEDIUM_HARD, HARD, VERY_HARD,
# competition, UNKNOWN_DIFFICULTY


def keep(difficulty: str) -> bool:
    if difficulty in KEEP_LABELS:
        return True
    try:
        return float(difficulty) <= MAX_NUMERIC
    except ValueError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1250, help="number of problems")
    ap.add_argument("--max-input-chars", type=int, default=4000)
    ap.add_argument("--max-source-frac", type=float, default=0.6,
                    help="cap for the dominant source (codeforces)")
    ap.add_argument("--stream-scan", type=int, default=60000,
                    help="max rows to scan before giving up on --n")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="ocr_problem_set.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("nvidia/OpenCodeReasoning", "split_0",
                      split="split_0", streaming=True)
    try:  # column projection: skip the huge `output` field while streaming
        ds = ds.select_columns(["id", "input", "source", "difficulty", "solution"])
    except Exception:
        pass

    rng = random.Random(args.seed)
    picked, seen, cap_cf = [], set(), int(args.n * args.max_source_frac)
    n_cf, scanned = 0, 0
    for r in ds:
        scanned += 1
        if scanned > args.stream_scan:
            break
        if r["difficulty"] not in (None, "") and not keep(r["difficulty"]):
            continue
        text = (r["input"] or "").strip()
        if not text or len(text) > args.max_input_chars or r["id"] in seen:
            continue
        if r["source"] == "codeforces":
            if n_cf >= cap_cf:
                continue
            n_cf += 1
        seen.add(r["id"])
        picked.append({"id": r["id"], "input": text,
                       "source": r["source"], "difficulty": r["difficulty"],
                       "solution": (r.get("solution") or "").strip()})
        if len(picked) >= args.n:
            break

    rng.shuffle(picked)
    n_out = 0
    with open(args.out, "w") as f:
        for p in picked:
            row = {"messages": [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": p["input"]},
                {"role": "assistant", "content": p["solution"]},
            ], "meta": {"source": f"opencodereasoning:{p['source']}",
                        "task_id": p["id"], "difficulty": p["difficulty"]}}
            f.write(json.dumps(row) + "\n")
            n_out += 1

    srcs = {}
    for p in picked:
        srcs[p["source"]] = srcs.get(p["source"], 0) + 1
    print(f"wrote {n_out} problems to {args.out} (scanned {scanned})")
    print("sources:", dict(sorted(srcs.items(), key=lambda kv: -kv[1])))
    return 0 if n_out == args.n else 1


if __name__ == "__main__":
    sys.exit(main())
