"""Invent NEW agent tasks + trajectories with a REMOTE model (OpenAI API).

Standalone: no ClearML task, no GPUs. This is the remote-model twin of
scripts/distill_data.py (make_remote_distill.py is the twin of code_distill.py).
Instead of loading the teacher into vLLM, both passes go to an
OpenAI-compatible endpoint:

  pass 1  invent tasks  -- seeded topic prompts (prompts.sample_task_prompt /
          sample_scicode_task_prompt) -> JSON task objects, validated and
          fingerprint-deduped (data.validate_task_obj / task_fingerprint)
  pass 2  trajectories  -- AGENT_SYSTEM_PROMPT + TRAJECTORY_INSTRUCTION ->
          JSONL action/observation transcript, validated
          (data.validate_transcript) with repair rounds (REPAIR_INSTRUCTION),
          exactly like the original distill

Outputs:
  <out>.jsonl              messages-format rows ready for
                           train_student.py --extra-data (rendered with
                           data.build_sft_rows, identical to how the trainer
                           renders the upstream distill data)
  <out>.transcripts.jsonl  raw distill-schema rows (title/repo_context/task/
                           fingerprint/kind/transcript) for archival or
                           re-rendering

Both are uploaded to S3; the printed --extra-data line points at the
messages-format file. Interrupted runs resume from journals in
~/.cache/remote_distill (pass 1 keyed by prompt index, pass 2 by task
fingerprint), so re-running costs only the unfinished work.

The student trains with --max-len 16384 and tail-truncates longer rows; this
script reports the rendered-row size distribution and can optionally drop
over-budget rows with --max-row-chars (default 0 = keep everything, same as
the original pipeline).

Examples
--------
  python scripts/make_remote_tasks.py \
      --url http://localhost:9090/v1 --token none --model org/model-a \
      --s3-uri s3://s3-bucket-loadbalancer.das-redpanda.production.svc.spin.nersc.org:5432/dasml/clearml/remote_distills/model-a_tasks.jsonl

  # smoke test: 12 task inventions instead of 1200
  python scripts/make_remote_tasks.py ... --n-tasks 12
"""

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import threading

try:
    from tqdm import tqdm
except ImportError:  # optional: progress falls back to periodic prints
    tqdm = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clearml_finetuning import data, prompts  # noqa: E402
from make_remote_distill import chat_remote, get_s3_ops  # noqa: E402


def upload(ops, local_path, uri):
    if uri.endswith("/"):
        uri += os.path.basename(local_path)
    return ops[1](local_path, uri)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(
        description="invent new agent tasks + trajectories with a remote "
                    "model and upload training JSONL to S3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("REMOTE_BASE_URL", ""),
                    help="OpenAI-compatible API base, e.g. http://localhost:9090/v1")
    ap.add_argument("--token", default=os.environ.get("REMOTE_API_TOKEN",
                    os.environ.get("OPENAI_API_KEY", "")),
                    help="bearer token for --url (env REMOTE_API_TOKEN)")
    ap.add_argument("--model", required=True, help="model name served at --url")
    ap.add_argument("--s3-uri", required=True,
                    help="destination for the messages-format JSONL, "
                         "s3://host:port/bucket/prefix/name.jsonl "
                         "(trailing / = prefix, basename appended); the "
                         "transcript file is uploaded next to it as "
                         "<name>.transcripts.jsonl")
    ap.add_argument("--s3-endpoint", default="",
                    help="override the endpoint derived from --s3-uri, e.g. http://host:9000")
    ap.add_argument("--s3-access-key", default="", help="or AWS_ACCESS_KEY_ID")
    ap.add_argument("--s3-secret-key", default="", help="or AWS_SECRET_ACCESS_KEY")
    ap.add_argument("--s3-session-token", default="", help="or AWS_SESSION_TOKEN")
    ap.add_argument("--s3-insecure", action="store_true",
                    help="talk plain HTTP to the S3 endpoint (boto3 path only; "
                         "for endpoints with secure:false in clearml.conf)")
    ap.add_argument("--out", default="",
                    help="local output basename without extension "
                         "(default: ./remote_tasks_<model>)")
    ap.add_argument("--cache", default="~/.cache/remote_distill",
                    help="cache dir for resume journals")
    ap.add_argument("--n-tasks", type=int, default=1200,
                    help="task inventions to ATTEMPT (validation + dedup reject "
                         "some; the original distill kept 1083/1200)")
    ap.add_argument("--scicode-frac", type=float, default=0.25,
                    help="fraction of task prompts that are SciCode-style "
                         "scientific problems")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--task-temp", type=float, default=0.9)
    ap.add_argument("--task-max-tokens", type=int, default=400)
    ap.add_argument("--traj-temp", type=float, default=0.7)
    ap.add_argument("--traj-max-tokens", type=int, default=2400)
    ap.add_argument("--repairs", type=int, default=2,
                    help="repair rounds for rejected trajectories")
    ap.add_argument("--max-row-chars", type=int, default=0,
                    help="drop final rows whose rendered messages exceed this "
                         "many characters (0 = keep everything; the trainer "
                         "tail-truncates over-budget rows at --max-len). "
                         "16384 tokens is roughly 57000 chars.")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=300, help="per-request seconds")
    ap.add_argument("--retry-kwargs", default="",
                    help="JSON merged into the request body for a RESCUE pass "
                         "over jobs whose first attempt returned empty; "
                         "empty = off (empties retry free on rerun)")
    args = ap.parse_args()

    if not args.url:
        raise SystemExit("--url required (or env REMOTE_BASE_URL)")
    if not args.token:
        raise SystemExit("--token required (or env REMOTE_API_TOKEN / OPENAI_API_KEY)")
    tag = args.model.replace("/", "_")
    base = args.out or f"remote_tasks_{tag}"
    out_msgs = base + ".jsonl"
    out_traj = base + ".transcripts.jsonl"
    # the transcript archive lands next to the messages file on S3
    msgs_uri = args.s3_uri
    if msgs_uri.endswith("/"):
        msgs_uri += os.path.basename(out_msgs)
    if msgs_uri.endswith(".jsonl"):
        traj_uri = msgs_uri[:-len(".jsonl")] + ".transcripts.jsonl"
    else:
        traj_uri = msgs_uri + ".transcripts.jsonl"
    # journals are keyed by model + generation identity: resuming with a
    # different seed/task count must never reuse old generations
    sig = hashlib.sha1(
        (args.model + f"|seed{args.seed}|n{args.n_tasks}"
         + f"|sci{args.scicode_frac}").encode()).hexdigest()[:10]
    cache = os.path.expanduser(args.cache)
    os.makedirs(cache, exist_ok=True)
    j1_path = os.path.join(cache, f"journal_tasks_{tag}_{sig}.jsonl")
    j2_path = os.path.join(cache, f"journal_traj_{tag}_{sig}.jsonl")

    ns_task = copy.copy(args)
    ns_task.temperature, ns_task.max_tokens = args.task_temp, args.task_max_tokens
    ns_traj = copy.copy(args)
    ns_traj.temperature, ns_traj.max_tokens = args.traj_temp, args.traj_max_tokens

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def gen_pool(ns, jobs, journal_path, journal, label,
                 extra_body=None, _top=True):
        """Run chat jobs {key: messages} not yet in the journal, concurrently.
        Appends {key, text} records; returns {key: text} including prior ones.

        On the outer call, if --retry-kwargs is set, a RESCUE pass re-asks the
        jobs that returned empty content with those kwargs merged into the
        request body (thinking models can ramble past any completion budget;
        bounding thinking, e.g. chat_template_kwargs {"thinking": false},
        recovers an answer)."""
        todo = {k: m for k, m in jobs.items() if k not in journal}
        print(f"{label}: {len(todo)} to generate ({len(journal)} already done)",
              flush=True)
        if not todo:
            return journal
        lock = threading.Lock()
        jf = open(journal_path, "a", encoding="utf-8")
        counters = {"ok": 0, "fail": 0}
        pbar = None if tqdm is None else tqdm(
            total=len(todo), desc=label, unit="task", dynamic_ncols=True,
            # disable=None: auto-hide when stderr is not a TTY (piped/nohup),
            # so log files get the plain print lines instead of bar spam
            disable=None)

        def work(key, msgs):
            return key, data.strip_think(chat_remote(ns, msgs, extra_body=extra_body))

        try:
            with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
                futs = {ex.submit(work, k, m): k for k, m in todo.items()}
                for n, fut in enumerate(as_completed(futs), 1):
                    key = futs[fut]
                    try:
                        key, text = fut.result()
                    except Exception as e:
                        counters["fail"] += 1
                        print(f"  [{n}/{len(todo)}] {key} FAILED: {e}", flush=True)
                    else:
                        if not text.strip():
                            counters["fail"] += 1
                            print(f"  [{n}/{len(todo)}] {key} empty generation",
                                  flush=True)
                        else:
                            with lock:
                                jf.write(json.dumps(
                                    {"key": key, "text": text,
                                     **({"rescued": True} if extra_body else {})},
                                    ensure_ascii=False) + "\n")
                                jf.flush()
                                journal[key] = text
                            counters["ok"] += 1
                    # every completion advances the bar (ok, empty, failed)
                    if pbar is not None and not pbar.disable:
                        pbar.update(1)
                        pbar.set_postfix(ok=counters["ok"], fail=counters["fail"])
                    elif n % 25 == 0 or n == len(todo):
                        # bar hidden (piped/nohup) or tqdm missing: plain prints
                        print(f"  [{n}/{len(todo)}] ok={counters['ok']} "
                              f"failed={counters['fail']}", flush=True)
        finally:
            if pbar is not None:
                pbar.close()
            jf.close()
        if _top and args.retry_kwargs:
            still = {k: m for k, m in jobs.items() if k not in journal}
            if still:
                extra = json.loads(args.retry_kwargs)
                print(f"{label} rescue: {len(still)} without completions, "
                      f"retrying with {args.retry_kwargs}", flush=True)
                journal = gen_pool(ns, still, journal_path, journal,
                                   f"{label} rescue", extra, _top=False)
        return journal

    def load_journal(path):
        done = {}
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["key"]] = rec["text"]
            print(f"resuming: {len(done)} records in {os.path.basename(path)}",
                  flush=True)
        return done

    # ---- pass 1: invent tasks ------------------------------------------------
    rng = random.Random(args.seed)
    n_scicode = int(round(args.n_tasks * args.scicode_frac))
    entries = []  # (kind, prompt) in the original distill's order
    for i in range(args.n_tasks):
        if i < n_scicode:
            entries.append(("scicode", prompts.sample_scicode_task_prompt(rng)))
        else:
            entries.append(("agent", prompts.sample_task_prompt(rng)))

    journal1 = gen_pool(ns_task, {str(i): [{"role": "user", "content": p}]
                                  for i, (_, p) in enumerate(entries)},
                        j1_path, load_journal(j1_path), "pass1 tasks")

    tasks, seen, rejects1, dups = [], set(), 0, 0
    for i, (kind, _) in enumerate(entries):
        text = journal1.get(str(i))
        if not text:
            continue
        obj = data.extract_json_object(text)
        cleaned = data.validate_task_obj(obj) if obj else None
        if not cleaned:
            rejects1 += 1
            continue
        fp = data.task_fingerprint(cleaned)
        if fp in seen:
            dups += 1
            continue
        seen.add(fp)
        cleaned["fingerprint"] = fp
        cleaned["kind"] = kind
        tasks.append(cleaned)
    print(f"pass1: kept {len(tasks)}/{len(entries)} invented tasks "
          f"({sum(1 for t in tasks if t['kind'] == 'scicode')} scicode-style, "
          f"{sum(1 for t in tasks if t['kind'] == 'agent')} agent; "
          f"{rejects1} rejected, {dups} duplicates)", flush=True)
    if not tasks:
        raise SystemExit("no tasks survived validation; nothing to do")

    # ---- pass 2: generate trajectories ---------------------------------------
    traj_jobs = {
        t["fingerprint"]: [
            {"role": "system", "content": prompts.AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompts.TRAJECTORY_INSTRUCTION.format(
                repo_context=t["repo_context"], task=t["task"])}]
        for t in tasks}
    journal2 = gen_pool(ns_traj, traj_jobs, j2_path,
                        load_journal(j2_path), "pass2 trajectories")

    rows, pending = [], []
    for t in tasks:
        text = journal2.get(t["fingerprint"])
        seg = data.validate_transcript(text) if text else None
        if seg:
            rows.append({**t, "transcript": seg})
        else:
            pending.append((t, text or ""))
    print(f"pass2 initial: kept {len(rows)}/{len(tasks)}, "
          f"{len(pending)} rejected -> repair", flush=True)

    # ---- pass 2b: repair rounds (each round repairs the latest broken text) ---
    for rnd in range(1, args.repairs + 1):
        if not pending:
            break
        repair_jobs = {
            f'{t["fingerprint"]}#r{rnd}': [
                {"role": "system", "content": prompts.AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": prompts.REPAIR_INSTRUCTION.format(
                    repo_context=t["repo_context"], task=t["task"],
                    broken=broken[:3000])}]
            for t, broken in pending}
        journal2 = gen_pool(ns_traj, repair_jobs, j2_path, journal2,
                            f"pass2 repair r{rnd}")
        still = []
        for t, _ in pending:
            text = journal2.get(f'{t["fingerprint"]}#r{rnd}')
            seg = data.validate_transcript(text) if text else None
            if seg:
                rows.append({**t, "transcript": seg})
            else:
                still.append((t, text or ""))
        print(f"repair round {rnd}: total kept {len(rows)}, "
              f"{len(still)} still rejected", flush=True)
        pending = still

    print(f"kept {len(rows)}/{len(tasks)} trajectories after validation",
          flush=True)
    if not rows:
        raise SystemExit("no trajectories survived validation; nothing to upload")

    # ---- assemble: messages rows for --extra-data + raw transcript archive ----
    rng.shuffle(rows)
    data.write_jsonl(out_traj, rows)
    msg_rows = data.build_sft_rows(rows, prompts.AGENT_SYSTEM_PROMPT)
    for mr, tr in zip(msg_rows, rows):  # preserve kind for downstream filtering
        mr["meta"]["kind"] = tr["kind"]
    dropped = 0
    if args.max_row_chars > 0:
        keep = [mr for mr in msg_rows
                if sum(len(m.get("content") or "") for m in mr["messages"])
                <= args.max_row_chars]
        dropped = len(msg_rows) - len(keep)
        msg_rows = keep
    data.write_jsonl(out_msgs, msg_rows)

    sizes = sorted(sum(len(m.get("content") or "") for m in mr["messages"])
                   for mr in msg_rows)
    p = lambda q: sizes[min(len(sizes) - 1, int(q * (len(sizes) - 1)))] if sizes else 0
    print(f"wrote {len(msg_rows)} training rows to {out_msgs} "
          f"(+ {len(rows)} raw transcripts to {out_traj}; dropped by "
          f"--max-row-chars: {dropped})", flush=True)
    print(f"rendered row size: p50={p(0.5)} p90={p(0.9)} p100={p(1.0)} chars "
          f"(~3.5 chars/token; student window 49152 tok ~= 172000 chars)", flush=True)

    # ---- upload ----------------------------------------------------------------
    ops = get_s3_ops(args)
    uri = upload(ops, out_msgs, msgs_uri)
    upload(ops, out_traj, traj_uri)
    print(f"\nuse in a train stage with:\n"
          f"  --extra-data {uri}\n"
          f"(raw transcripts uploaded to {traj_uri}; mix several sets by "
          f"comma-separating URLs in --extra-data)", flush=True)


if __name__ == "__main__":
    main()
