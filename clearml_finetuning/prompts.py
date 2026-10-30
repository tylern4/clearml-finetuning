"""Agent scaffold, prompt templates, and seed diversity for distillation."""

AGENT_SYSTEM_PROMPT = """\
You are a coding agent working inside a git repository on a Linux machine.

You act by emitting one JSON action per line (JSONL). Available actions:
  {"action": "run_cmd", "cmd": "<shell command>"}
  {"action": "read_file", "path": "<path>"}
  {"action": "edit_file", "path": "<path>", "old": "<exact text to replace>", "new": "<replacement>"}
  {"action": "write_file", "path": "<path>", "content": "<full file content>"}
  {"action": "finish", "summary": "<what you did and how you verified it>"}

Rules:
- Emit exactly one action per line, each a valid JSON object.
- After each action you will receive an observation; use it to decide the next step.
- Verify your work (run tests / the relevant commands) before finishing.
- Keep actions minimal and precise; do not restate file contents unnecessarily.
"""

SEED_TOPICS = [
    # languages / runtimes
    "python", "typescript", "go", "rust", "bash", "sql", "java", "c++",
    # frameworks / tooling
    "pytest", "flask", "fastapi", "django", "react", "docker", "kubernetes",
    "github-actions", "make", "cmake", "numpy", "pandas", "pytorch",
    # bug / task classes
    "off-by-one error", "race condition", "resource leak", "null handling",
    "encoding bug", "regex fix", "api pagination", "error handling",
    "performance regression", "flaky test", "dependency upgrade",
    "deprecation warning", "type error", "memory growth", "deadlock",
    "config parsing", "log formatting", "cli flag handling", "data validation",
    # task archetypes
    "fix failing unit test", "add missing test coverage", "fix a crash on edge case",
    "add a small feature", "refactor duplicated code", "improve error message",
    "optimize a hot loop", "write a migration script", "fix broken import",
    "add input validation", "fix timezone handling", "make a retry idempotent",
]

TASK_GEN_INSTRUCTION = """\
Invent a realistic, small, self-contained software engineering task.

Topic area: {topic}
Task type: {archetype}

Return ONLY a JSON object with keys:
  "title": short task title
  "repo_context": 1-2 sentences describing the (fictional) repo and relevant files
  "task": the task description as a maintainer would write it (2-5 sentences,
          concrete, mentioning file names and expected behavior)
The task must be solvable with a handful of file edits and shell commands.
"""

TRAJECTORY_INSTRUCTION = """\
{repo_context}

Task: {task}

You are the coding agent described in the system prompt. Produce the full
action-by-action solution as JSONL: each line is one action object, starting
with read/run actions to inspect the relevant files, then edit/write actions,
then a verification run_cmd (e.g. running the test), and finally a finish
action with a one-paragraph summary. Keep it realistic and terse: 4-10 actions.
The tool outputs are not shown; write only your actions.
"""


def sample_task_prompt(rng):
    """Build a task-generation prompt from a random topic/archetype pair."""
    import random

    topic = rng.choice(SEED_TOPICS)
    archetype = rng.choice(SEED_TOPICS[SEED_TOPICS.index("fix failing unit test"):])
    return TASK_GEN_INSTRUCTION.format(topic=topic, archetype=archetype)
