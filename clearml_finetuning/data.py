"""Data helpers: trajectory schema, validation, and chat formatting."""

import hashlib
import json
import os
import re

_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)


def strip_think(text):
    """Remove reasoning traces (<think ...>...</think) from model output."""
    if not isinstance(text, str):
        return text
    return _THINK_RE.sub("", text).strip()


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
    text = strip_think(text)
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
    text = strip_think(text)
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


ACTION_KEYS = {"run_cmd", "read_file", "edit_file", "write_file", "finish"}


def validate_transcript(obj):
    """Validate an interleaved action/observation transcript.

    Returns a list of segments: {"kind": "action"|"observation", ...} or None.
    Must start with an action, end with a finish action, contain >= 3 actions
    including at least one edit/write and one run_cmd.
    """
    lines = extract_jsonl_actions(obj)
    if not lines:
        return None
    segments = []
    n_actions = 0
    for line in lines:
        try:
            obj_line = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj_line, dict):
            return None
        if "action" in obj_line:
            if obj_line["action"] not in ACTION_KEYS:
                return None
            segments.append({"kind": "action", "data": obj_line})
            n_actions += 1
        elif "observation" in obj_line:
            obs = obj_line["observation"]
            if not isinstance(obs, str) or not (1 <= len(obs) <= 2000):
                return None
            # observations must follow an action (never start the transcript)
            if not segments or segments[-1]["kind"] != "action":
                return None
            segments.append({"kind": "observation", "text": obs})
        else:
            return None
    if not (3 <= n_actions <= 8):
        return None
    if not segments or segments[-1]["kind"] != "action" \
            or segments[-1]["data"]["action"] != "finish":
        return None
    kinds = [s_["data"]["action"] for s_ in segments if s_["kind"] == "action"]
    if "edit_file" not in kinds and "write_file" not in kinds:
        return None
    if "run_cmd" not in kinds:
        return None
    return segments


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


def normalize_messages(messages):
    """Enforce chat-template invariants: at most one system message, in slot 0.

    Some sources (e.g. cyber-security conversations) carry their own system
    turns; msg_row/converters prepend another one, and Qwen-style templates
    reject any system message that is not the very first message ("System
    message must be at the beginning"). Later system turns are demoted to
    user turns, merged into an adjacent user turn when possible.
    """
    out = []
    for m in messages:
        if m.get("role") == "system" and out:
            content = m.get("content", "")
            if out[-1].get("role") == "user":
                out[-1]["content"] += "\n\n[context]\n" + content
            else:
                out.append({"role": "user", "content": "[context]\n" + content})
            continue
        out.append(m)
    return out


def build_sft_rows(rows, system_prompt):
    """Convert distilled transcript rows into multi-turn chat training rows.

    Actions become assistant turns; simulated observations become user turns
    (they are environment output and get masked by assistant-only loss).
    """
    import json as _json

    out = []
    for row in rows:
        task_text = f"{row['repo_context']}\n\nTask: {row['task']}"
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": task_text}]
        for seg in row["transcript"]:
            if seg["kind"] == "action":
                messages.append({"role": "assistant",
                                 "content": _json.dumps(seg["data"], ensure_ascii=False)})
            else:
                messages.append({"role": "user",
                                 "content": f"Observation:\n{seg['text']}"})
        out.append({
            "messages": messages,
            "meta": {"title": row["title"], "fingerprint": row["fingerprint"]},
        })
    return out
