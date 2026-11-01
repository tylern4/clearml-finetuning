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


# SciCode-style variation: the teacher invents NEW research-level scientific
# coding problems (in SciCode's domains) and then produces agent trajectories
# on them. These are fresh problems, NOT the SciCode test split, so there is
# no evaluation contamination.
SCICODE_DOMAINS = [
    "numerical linear algebra (matrix decompositions, eigenvalues, linear solves)",
    "ordinary/partial differential equations (integrators, solvers)",
    "numerical integration and quadrature",
    "optimization and root finding",
    "probability and statistics (sampling, distributions, Monte Carlo)",
    "signal processing (FFT, filtering, spectral analysis)",
    "classical mechanics and molecular dynamics",
    "electromagnetism and electrostatics (field solvers)",
    "quantum mechanics (Schrodinger equation, tight-binding, Hartree-Fock)",
    "statistical mechanics and Ising/Potts models",
    "quantum chemistry (basis sets, integrals, SCF)",
    "condensed matter physics (band structure, phonons)",
    "thermodynamics and phase transitions",
    "chemical kinetics and reaction networks",
    "bioinformatics (sequence alignment, phylogenetics)",
    "materials science (lattice structures, elastic constants)",
    "astrophysics (N-body, orbital mechanics, radiative transfer)",
    "fluid dynamics (Navier-Stokes, lattice Boltzmann)",
]

SCICODE_TASK_GEN_INSTRUCTION = """\
Invent a realistic, self-contained scientific computing task suitable for a
research code base. Do NOT copy any existing benchmark problem.

Scientific domain: {domain}
Problem archetype: {archetype}

The task should require writing or fixing a small Python function (or a few
functions) using scientific libraries (numpy/scipy/sympy/etc.).

Return ONLY a JSON object with keys:
  "title": short task title
  "repo_context": 1-2 sentences describing the (fictional) research repo and
                  the relevant module/file names
  "task": the task description as a researcher would write it (2-5 sentences,
          concrete, naming the file, the function signature to implement, and
          the expected numerical behavior / units / tolerances)
The task must be solvable with a handful of file edits and shell commands
(e.g. running a small check script with numpy).
"""

SCICODE_ARCHETYPES = [
    "implement a missing numerical function from its spec and docstring",
    "fix an incorrect result caused by a wrong formula or sign",
    "fix a numerical instability (overflow, divide-by-zero, poor conditioning)",
    "add a convergence/accuracy check and a unit test with tolerances",
    "vectorize a slow pure-Python loop with numpy",
    "fix an off-by-one / indexing bug in an array computation",
    "implement a boundary or initial condition that is currently wrong",
    "add input validation for physical parameters (negative mass, etc.)",
]


def sample_scicode_task_prompt(rng):
    """Build a SciCode-style scientific-task generation prompt."""
    domain = rng.choice(SCICODE_DOMAINS)
    archetype = rng.choice(SCICODE_ARCHETYPES)
    return SCICODE_TASK_GEN_INSTRUCTION.format(domain=domain, archetype=archetype)


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


REPAIR_INSTRUCTION = """\
{repo_context}

Task: {task}

Your previous transcript for this task was rejected because it did not follow
the required JSONL schema. Here is what you produced:

---BEGIN BROKEN TRANSCRIPT---
{broken}
---END BROKEN TRANSCRIPT---

Rewrite it as a CORRECT transcript, JSONL only, obeying ALL of these rules:
- One JSON object per line, strictly alternating: action, observation, action, ...
- Action lines use exactly one of these forms:
  {{"action": "run_cmd", "cmd": "<shell command>"}}
  {{"action": "read_file", "path": "<path>"}}
  {{"action": "edit_file", "path": "<path>", "old": "<exact text>", "new": "<replacement>"}}
  {{"action": "write_file", "path": "<path>", "content": "<full file content>"}}
  {{"action": "finish", "summary": "<what you did and how you verified>"}}
- Every action except the final finish MUST be immediately followed by an
  observation line: {{"observation": "<short realistic tool output, 1-2000 chars>"}}
- The transcript MUST start with an action (never an observation).
- 3 to 8 actions total.
- MUST end with a finish action (no observation after finish).
- MUST include at least one edit_file or write_file AND at least one run_cmd.
Output the corrected JSONL only - no commentary, no markdown fences.
"""
