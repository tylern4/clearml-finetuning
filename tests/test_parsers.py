"""Offline checks of the distillation parsers (no network, no clearml)."""
import sys

sys.path.insert(0, ".")
from clearml_finetuning import data  # noqa: E402

TRAJ = """Here are the actions:
```jsonl
{"action": "run_cmd", "cmd": "ls tests"}
{"action": "edit_file", "path": "a.py", "old": "x = 1", "new": "x = 2"}
{"action": "read_file", "path": "a.py"}
{"action": "run_cmd", "cmd": "pytest -x"}
{"action": "finish", "summary": "fixed and verified"}
```
<|im_end|>"""

TASK = """Sure! Here is a task:
```json
{"title": "Fix login bug on refresh", "repo_context": "A web app with session handling and cookie TTLs.",
 "task": "Users get logged out on refresh. Fix the session cookie TTL handling so refresh keeps the session."}
```
"""

acts = data.validate_actions(TRAJ)
assert acts is not None, "trajectory parse failed"
assert len(acts) == 5, f"expected 5 actions, got {len(acts)}"
assert acts[-1]["action"] == "finish"
print(f"trajectory: OK ({len(acts)} actions)")

obj = data.extract_json_object(TASK)
cleaned = data.validate_task_obj(obj) if obj else None
assert cleaned, "task parse failed"
print(f"task: OK (title={cleaned['title']!r})")

# degenerate cases must NOT pass
assert data.validate_actions("no actions here") is None
assert data.validate_actions('{"action": "finish"}') is None
assert data.validate_task_obj({"title": "x"}) is None
print("negatives: OK")
