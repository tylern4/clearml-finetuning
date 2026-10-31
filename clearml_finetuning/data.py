"""Data helpers: trajectory schema, validation, and chat formatting."""

import hashlib
import json
import os


def trajectory_to_messages(system_prompt, task_text, actions):
    """Render a distilled trajectory as a chat message list."""
    import json as _json

    body = "\n".join(_json.dumps(a, ensure_ascii=False) for a in actions)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_text},
        {"role": "assistant", "content": body},
    ]


def extract_json_object(text):
    """Best-effort extraction of a JSON object from LLM output.

    Handles prose around the object, fences, and stray stop tokens by slicing
    from the first '{' to the last '}'.
    """
    if not isinstance(text, str):
        return None
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return None


def extract_jsonl_actions(text):
    """Extract JSONL action lines from LLM output.

    Tolerates markdown fences and prose: only lines starting with '{' are
    considered, which naturally skips fence markers and commentary.
    """
    if not isinstance(text, str):
        return []
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    return [line.strip() for line in text.splitlines() if line.strip().startswith("{")]


def validate_task_obj(obj):
    """Return a cleaned task dict or None if the teacher output is unusable."""
    if not isinstance(obj, dict):
        return None
    title, ctx, task = obj.get("title"), obj.get("repo_context"), obj.get("task")
    if not isinstance(title, str) or not isinstance(task, str):
        return None
    ctx = ctx if isinstance(ctx, str) else ""
    if not (10 <= len(title) <= 160 and len(ctx) <= 800 and 40 <= len(task) <= 1500):
        return None
    return {"title": title.strip(), "repo_context": ctx.strip(), "task": task.strip()}


def validate_actions(obj):
    """Parse/validate a teacher trajectory; returns a list of action dicts or None."""
    if isinstance(obj, dict):  # already-parsed single action is not a trajectory
        return None
    lines = extract_jsonl_actions(obj)
    if not lines:
        return None
    actions = []
    allowed = {"run_cmd", "read_file", "edit_file", "write_file", "finish"}
    for line in lines:
        try:
            a = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(a, dict) or a.get("action") not in allowed:
            return None
        actions.append(a)
    if not (4 <= len(actions) <= 14):
        return None
    if actions[-1].get("action") != "finish":
        return None
    if not any(a["action"] in ("edit_file", "write_file") for a in actions):
        return None
    if not any(a["action"] == "run_cmd" for a in actions):
        return None
    return actions


def task_fingerprint(task_obj):
    """Stable hash for dedup."""
    blob = json.dumps(task_obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_sft_rows(rows, system_prompt):
    """Convert distilled rows into chat-message training rows."""
    out = []
    for row in rows:
        task_text = f"{row['repo_context']}\n\nTask: {row['task']}"
        out.append({
            "messages": trajectory_to_messages(system_prompt, task_text, row["actions"]),
            "meta": {"title": row["title"], "fingerprint": row["fingerprint"]},
        })
    return out
