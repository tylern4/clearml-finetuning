"""Stage: prepare extra code-domain SFT data from Hugging Face datasets.

Sources (each converted to {"messages": [...]} SFT rows):
  - bigcode/bigcodebench (public)        function-level code completion
  - harborframework/terminal-bench-science (public)
        terminal-bench tasks; the oracle solution/solve.sh is rendered as an
        agent-scaffold JSONL trajectory (run_cmd actions + finish)
  - SALT-NLP/SWE-chat (GATED)            real coding-agent conversations;
        needs an HF token that has accepted the gate (HF_TOKEN env)
  - Vyber07/cyber-security (GATED)       cybersecurity Q&A in ShareGPT /
        Instruction-IO formats; also needs HF_TOKEN

Gated sources are skipped with a warning when no usable token is present, so
the stage still succeeds with the public datasets only.

Output artifact `code-data`: a directory containing code_train.jsonl.
"""

import argparse
import json
import os
import random
import sys
import urllib.request

# gated datasets need an HF token: huggingface_hub reads HF_TOKEN or
# $HOME/.cache/huggingface/token; also accept the $HOME/.hf_token convention
# (on the compute nodes $HOME is the worker's home, not the login home)
if not os.environ.get("HF_TOKEN"):
    from pathlib import Path

    _home = Path.home()
    for _tok_file in (_home / ".hf_token",                       # $HOME/.hf_token (muller dasml home)
                      _home / ".cache" / "huggingface" / "token",
                      Path("/mscratch/sd/d/dasml/.hf_token"),    # worker container HOME=/root; dasml home is on mscratch
                      Path("/mscratch/sd/d/dasml/.cache/huggingface/token")):
        if _tok_file.exists():
            os.environ["HF_TOKEN"] = _tok_file.read_text().strip()
            print(f"HF_TOKEN loaded from {_tok_file}")
            break
    else:
        print(f"no HF token found (looked in {_home}/.hf_token etc.) - "
              f"gated datasets will be skipped")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clearml_finetuning import cml, data, prompts  # noqa: E402

BCB_REPO = "bigcode/bigcodebench"
BCB_PARQUET = "data/v0.1.4-00000-of-00001.parquet"
TBS_REPO = "harborframework/terminal-bench-science"
SWE_REPO = "SALT-NLP/SWE-chat"
CYBER_REPO = "Vyber07/cyber-security"

MAX_MSG_CHARS = 4000


def cap_text(t, limit=MAX_MSG_CHARS):
    if not isinstance(t, str):
        return ""
    return t if len(t) <= limit else t[:limit] + "\n... [truncated]"


def msg_row(system, messages, meta):
    messages = [{"role": m["role"], "content": cap_text(m["content"])}
                for m in messages if m.get("content")]
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return None
    return {"messages": [{"role": "system", "content": system}] + messages,
            "meta": meta}


# ---------------------------------------------------------------- bigcodebench
def bigcodebench_rows(cap):
    from huggingface_hub import hf_hub_download
    import pandas as pd

    path = hf_hub_download(BCB_REPO, BCB_PARQUET, repo_type="dataset")
    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        libs = r.get("libs") or []
        user = f"{r['code_prompt']}\n\n{r['instruct_prompt']}"
        if libs:
            user += f"\n\nAllowed libraries: {', '.join(libs)}"
        assistant = f"{r['complete_prompt']}{r['canonical_solution']}"
        row = msg_row(prompts.CODE_SYSTEM_PROMPT, [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ], {"source": "bigcodebench", "task_id": str(r.get("task_id", ""))})
        if row:
            rows.append(row)
        if len(rows) >= cap:
            break
    return rows


# ------------------------------------------------------- terminal-bench-science
def _ds_rows_api(repo, config, split, length=100):
    """Fetch rows via the public datasets-server API (no auth needed)."""
    url = ("https://datasets-server.huggingface.co/rows?dataset="
           f"{urllib.request.quote(repo, safe='')}&config={config}&split={split}"
           f"&offset=0&length={length}")
    with urllib.request.urlopen(url, timeout=120) as r:
        j = json.loads(r.read())
    if j.get("error"):
        raise RuntimeError(j["error"])
    return j["rows"]


def tbs_rows(cap):
    """Render each task's oracle solve.sh as an agent JSONL trajectory."""
    from huggingface_hub import hf_hub_download

    rows = []
    for entry in _ds_rows_api(TBS_REPO, "default", "test", length=100):
        p = entry["row"]
        loc = p.get("location") or ""
        if not loc:
            continue
        try:
            # `location` already includes the tasks/ prefix
            solve = hf_hub_download(TBS_REPO, f"{loc}/solution/solve.sh",
                                    repo_type="dataset")
            with open(solve, encoding="utf-8", errors="replace") as f:
                script = f.read()
        except Exception as e:
            print(f"  tbs {p['name']}: no solve.sh ({e}); skipping")
            continue
        actions = []
        for line in script.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            actions.append({"action": "run_cmd", "cmd": line})
        actions.append({"action": "finish",
                        "summary": f"Solved {p['name']} per the oracle solution."})
        import json as _json
        body = "\n".join(_json.dumps(a, ensure_ascii=False) for a in actions)
        user = (f"{p['instruction']}\n\nThe task environment is already set up "
                f"in /app. Solve the task using shell commands and file edits.")
        row = msg_row(prompts.AGENT_SYSTEM_PROMPT, [
            {"role": "user", "content": user},
            {"role": "assistant", "content": body},
        ], {"source": "terminal-bench-science", "task": p["name"]})
        if row:
            rows.append(row)
        if len(rows) >= cap:
            break
    return rows


# ------------------------------------------------------------------- SWE-chat
def swechat_rows(cap, seed=7):
    """Sample real coding-agent conversations (is_conversational turns only)."""
    from datasets import load_dataset

    ds = load_dataset(SWE_REPO, "conversations", split="train", streaming=True)
    rng = random.Random(seed)
    by_session, picked = {}, []
    for ex in ds:
        if not ex.get("is_conversational"):
            continue
        sid = ex.get("session_id")
        role = "user" if ex.get("turn_type") == "user_prompt" else "assistant"
        content = cap_text(ex.get("content") or "")
        if not content:
            continue
        by_session.setdefault(sid, []).append({"role": role, "content": content})
        if len(by_session) >= cap * 4 and len(picked) < cap:
            break
    sessions = list(by_session.items())
    rng.shuffle(sessions)
    for sid, msgs in sessions:
        # keep at most 8 conversational messages, starting at a user turn
        msgs = msgs[:8]
        if msgs and msgs[0]["role"] != "user":
            msgs = msgs[1:]
        row = msg_row(prompts.CODING_ASSISTANT_SYSTEM_PROMPT, msgs,
                      {"source": "swe-chat", "session_id": sid})
        if row:
            picked.append(row)
        if len(picked) >= cap:
            break
    return picked


# --------------------------------------------------------------- cyber-security
def cybersec_rows(cap, seed=7):
    """Sample records from the ShareGPT / Instruction-IO JSON files."""
    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files(CYBER_REPO, repo_type="dataset")
    json_files = [f for f in files
                  if f.endswith(".json") and f.count("/") <= 1]
    rng = random.Random(seed)
    rng.shuffle(json_files)
    rows = []
    for f in json_files:
        if len(rows) >= cap:
            break
        try:
            path = hf_hub_download(CYBER_REPO, f, repo_type="dataset")
            with open(path, encoding="utf-8", errors="replace") as fh:
                rec = json.load(fh)
        except Exception as e:
            print(f"  cybersec {f}: skipped ({e})")
            continue
        msgs = None
        meta = {"source": "cyber-security", "file": f}
        if isinstance(rec, dict) and isinstance(rec.get("conversations"), list):
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            msgs = [{"role": role_map.get(c.get("from"), "user"),
                     "content": c.get("value", "")}
                    for c in rec["conversations"]]
            msgs = [m for m in msgs if m["content"]]
            msgs = msgs[:6]
        elif isinstance(rec, dict) and rec.get("Instruction") and rec.get("Output"):
            user = rec["Instruction"]
            if rec.get("Input"):
                user += f"\n\nInput: {rec['Input']}"
            msgs = [{"role": "user", "content": user},
                    {"role": "assistant", "content": rec["Output"]}]
        if msgs:
            row = msg_row(prompts.CYBERSEC_SYSTEM_PROMPT, msgs, meta)
            if row:
                rows.append(row)
    return rows


def try_source(name, fn):
    try:
        rows = fn()
        print(f"  {name}: {len(rows)} rows")
        return rows
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "gated" in msg.lower() or "auth" in msg.lower():
            print(f"  {name}: SKIPPED (gated - set HF_TOKEN with accepted gate)")
        else:
            print(f"  {name}: FAILED ({msg[:200]})")
        return []


def main() -> None:
    ap = argparse.ArgumentParser(description="prepare extra code-domain SFT data")
    ap.add_argument("--out-dir", default="code")
    ap.add_argument("--max-bigcodebench", type=int, default=1140)
    ap.add_argument("--max-tbs", type=int, default=1000)
    ap.add_argument("--max-swechat", type=int, default=300)
    ap.add_argument("--max-cybersec", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    run = cml.get_run_logger("code_data", vars(args))
    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []
    all_rows += try_source("bigcodebench", lambda: bigcodebench_rows(args.max_bigcodebench))
    all_rows += try_source("terminal-bench-science", lambda: tbs_rows(args.max_tbs))
    all_rows += try_source("swe-chat", lambda: swechat_rows(args.max_swechat, args.seed))
    all_rows += try_source("cyber-security", lambda: cybersec_rows(args.max_cybersec, args.seed))

    random.Random(args.seed).shuffle(all_rows)
    out_path = os.path.join(args.out_dir, "code_train.jsonl")
    data.write_jsonl(out_path, all_rows)
    by_src = {}
    for r in all_rows:
        by_src[r["meta"].get("source", "?")] = by_src.get(r["meta"].get("source", "?"), 0) + 1
    print(f"wrote {len(all_rows)} rows to {out_path}  by source: {by_src}")

    if run:
        run.log({f"code_data/{k}": v for k, v in by_src.items()}, step=0)
        run.log({"code_data/total": len(all_rows)}, step=0)
        from pathlib import Path
        run.task.upload_artifact("code-data", Path(args.out_dir), wait_on_upload=True)
        run.finish()


if __name__ == "__main__":
    main()
