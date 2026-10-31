"""Offline checks of the distillation parsers (no network, no clearml)."""
import sys

sys.path.insert(0, ".")
from clearml_finetuning import data  # noqa: E402

TRAJ = """Here are the actions:
```jsonl
{"action": "run_cmd", "cmd": "ls tests"}
{"observation": "a.py  b.py  test_a.py"}
{"action": "edit_file", "path": "a.py", "old": "x = 1", "new": "x = 2"}
{"observation": "edited a.py"}
{"action": "run_cmd", "cmd": "pytest -x"}
{"observation": "1 passed"}
{"action": "finish", "summary": "fixed and verified"}
```
<|im_end|>"""

TASK = """Sure! Here is a task:
```json
{"title": "Fix login bug on refresh", "repo_context": "A web app with session handling and cookie TTLs.",
 "task": "Users get logged out on refresh. Fix the session cookie TTL handling so refresh keeps the session."}
```
"""

segs = data.validate_transcript(TRAJ)
assert segs is not None, "trajectory parse failed"
acts = [s["data"] for s in segs if s["kind"] == "action"]
assert len(acts) == 4, f"expected 4 actions, got {len(acts)}"
assert acts[-1]["action"] == "finish"
print(f"trajectory: OK ({len(acts)} actions)")

obj = data.extract_json_object(TASK)
cleaned = data.validate_task_obj(obj) if obj else None
assert cleaned, "task parse failed"
print(f"task: OK (title={cleaned['title']!r})")

# degenerate cases must NOT pass
assert data.validate_transcript("no actions here") is None
assert data.validate_transcript('{"action": "finish"}') is None
assert data.validate_task_obj({"title": "x"}) is None
print("negatives: OK")
