"""Agent scaffold, prompt templates, and seed diversity for distillation."""

AGENT_SYSTEM_PROMPT = """\
You are a coding agent working inside a git repository on a Linux machine.

You act by emitting JSONL lines. There are two kinds of lines:

1. Action lines (yours):
  {"action": "run_cmd", "cmd": "<shell command>"}
  {"action": "read_file", "path": "<path>"}
  {"action": "edit_file", "path": "<path>", "old": "<exact text to replace>", "new": "<replacement>"}
  {"action": "write_file", "path": "<path>", "content": "<full file content>"}
  {"action": "finish", "summary": "<what you did and how you verified it>"}

2. Observation lines (from the environment, right after each action):
  {"observation": "<what the tool returned: command output, file excerpt, or error>"}

Rules:
- One JSON object per line, alternating: action, its observation, next action, ...
- After each action, write the observation the tool would plausibly return
  (realistic command output or file content excerpt; keep it short).
- Verify your work (run the tests / relevant commands) before finishing.
- The transcript ends with a finish action (no observation after finish).
- Keep actions minimal and precise.
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

Produce the COMPLETE transcript for this task as JSONL, exactly following
the system prompt schema: for every action you take, include the observation
line the tool would plausibly return (realistic command output or a short
file excerpt), then the next action, and so on. Start with read/run actions
to inspect the relevant files, make the edit/write, verify with run_cmd,
and end with a finish action summarizing the change. 3-6 actions total.
Output the JSONL only - no commentary, no markdown fences.
"""


def sample_task_prompt(rng):
    """Build a task-generation prompt from a random topic/archetype pair."""
    import random

    topic = rng.choice(SEED_TOPICS)
    archetype = rng.choice(SEED_TOPICS[SEED_TOPICS.index("fix failing unit test"):])
    return TASK_GEN_INSTRUCTION.format(topic=topic, archetype=archetype)


SCICODE_SYSTEM_PROMPT = """\
You are a scientific coding assistant. You solve research-level problems from
mathematics, physics, chemistry, biology, and materials science by writing
correct, self-contained Python functions.

Rules:
- Implement exactly the function signature given; do not rename it.
- Use only the listed dependencies; import them inside the function body or
  at the top of your answer as needed.
- Output only the Python code - no commentary, no markdown fences.
"""

SCICODE_USER_TEMPLATE = """\
{problem_description}

Dependencies you may use:
{dependencies}

{step_description}

Complete the following function:
{function_header}"""
