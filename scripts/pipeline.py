"""Pipeline controller for clearml-finetuning.

Creates, enqueues, and polls the distill -> train -> eval stage tasks on the
configured ClearML queue. Requires the repo to be pushed: each task's script
section is pinned to the GitHub repo at the current branch head commit, so the
agent always clones the source of truth.

Usage:
    python scripts/pipeline.py --all
    python scripts/pipeline.py --only distill
    python scripts/pipeline.py --from train --upstream-distill <task_id>
"""

import argparse
import os
import re
import subprocess
import sys
import time

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUEUE = os.environ.get("CLEARML_QUEUE", "muller")
DOCKER = os.environ.get("CML_DOCKER", "nvcr.io/nvidia/cuda:13.1.1-runtime-ubuntu24.04")
REPO_URL = os.environ.get("CML_REPO_URL", "https://github.com/tylern4/clearml-finetuning.git")
BRANCH = os.environ.get("CML_REPO_BRANCH", "main")
TERMINAL = {"completed", "failed", "stopped"}

# per-stage venv isolation: vllm and peft stacks must not share a venv
REQUIREMENTS = {
    "distill": ["vllm>=0.11.0", "ninja", "nvidia-cuda-nvrtc==13.0.88",
                "clearml>=2.1", "boto3", "matplotlib"],
    "scicode_data": ["clearml>=2.1", "boto3", "gdown", "h5py", "numpy"],
    "code_data": ["clearml>=2.1", "boto3", "huggingface_hub", "datasets",
                  "pandas", "pyarrow"],
    "train": ["torch>=2.9.0", "transformers>=4.57", "peft>=0.17", "accelerate>=1.10",
              "bitsandbytes>=0.46", "clearml>=2.1", "boto3", "matplotlib"],
    "eval": ["vllm>=0.11.0", "ninja", "nvidia-cuda-nvrtc==13.0.88", "peft>=0.17",
             "h5py", "numpy", "scipy", "clearml>=2.1", "boto3", "matplotlib"],
    "convert_gguf": ["torch>=2.9.0", "transformers>=4.57", "peft>=0.17",
                     "gguf>=0.13", "huggingface_hub", "clearml>=2.1", "boto3"],
}
STAGE_TIMEOUTS = {"distill": 6 * 3600, "scicode_data": 2 * 3600, "code_data": 2 * 3600,
                  "train": 12 * 3600,
                  "eval": 8 * 3600, "convert_gguf": 3 * 3600}


def git_head():
    out = subprocess.run(["git", "rev-parse", f"origin/{BRANCH}"],
                         cwd=REPO_DIR, capture_output=True, text=True)
    return out.stdout.strip()


def submit(stage_args: list) -> str:
    """Run the stage runner locally until it self-enqueues; return the task id."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CLEARML_", "TRAINS_", "WF_"))}
    env["WF_OUTPUT_URI"] = os.environ.get("WF_OUTPUT_URI", "")
    out = subprocess.run(
        [sys.executable, "-u", "-m", "scripts.clearml_stage", *stage_args],
        capture_output=True, text=True, timeout=600,
        cwd=REPO_DIR, env=env,
    )
    m = re.search(r"new task id=([0-9a-f]{32})", out.stdout + out.stderr)
    if not m:
        raise RuntimeError(f"submit failed for {stage_args}\n"
                           f"STDOUT:\n{out.stdout[-1200:]}\nSTDERR:\n{out.stderr[-1200:]}")
    return m.group(1)


def finalize(task_id: str, stage: str):
    """Enqueue. The stage runner already pinned repo/commit/requirements/docker
    before its own execute_remotely enqueue; re-apply here best-effort in case
    the task is still editable (a fast worker may already have grabbed it, in
    which case the server rejects edits with ValueError and we just proceed)."""
    from clearml import Task

    t = Task.get_task(task_id=task_id)
    try:
        t.reset()
        commit = git_head()
        if not commit:
            raise RuntimeError("cannot resolve origin/{branch} - push the repo first")
        t._edit(script={
            "repository": REPO_URL,
            "branch": BRANCH,
            "version_num": commit,
            "working_dir": ".",
            "entry_point": "scripts/clearml_stage.py",
            "requirements": {"pip": REQUIREMENTS[stage]},
        })
        t.set_base_docker(docker_image=DOCKER)
        t.flush()
    except ValueError:
        print(f"  task {task_id} already picked up by a worker; skipping re-pin")
    from clearml import Task as T

    T.enqueue(t, queue_name=QUEUE)
    return t


def wait_for(task_id: str, timeout_s: float) -> str:
    from clearml import Task

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st = Task.get_task(task_id=task_id).status
        if st in TERMINAL:
            return st
        time.sleep(30)
    return "timeout"


def run_stage(name: str, stage_args: list) -> str:
    print(f"== stage {name} ==")
    tid = submit(stage_args)
    finalize(tid, name)
    print(f"  task {tid} enqueued to '{QUEUE}'")
    st = wait_for(tid, STAGE_TIMEOUTS[name])
    print(f"  stage {name}: {st}  ({tid})")
    if st != "completed":
        raise RuntimeError(f"stage {name} ended with {st}")
    return tid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["distill", "scicode_data", "code_data", "train",
                                       "eval", "convert_gguf"])
    ap.add_argument("--from", dest="from_stage",
                    choices=["code_data", "train", "eval", "convert_gguf"])
    ap.add_argument("--upstream-distill", default="")
    ap.add_argument("--upstream-train", default="")
    ap.add_argument("--upstream-scicode", default="")
    ap.add_argument("--upstream-code", default="")
    ap.add_argument("--resume-train", default="",
                    help="previous train task id to resume LoRA from (chunked training)")
    ap.add_argument("--epoch-offset", type=int, default=0,
                    help="epoch seed offset passed to the train stage (for resumed chunks)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--epochs", type=int, default=20,
                    help="training epochs passed to the train stage")
    args = ap.parse_args()

    t0 = time.time()
    ids = {}
    distill_args = ["--stage", "distill"]
    scicode_args = ["--stage", "scicode_data"]
    code_args = ["--stage", "code_data"]
    train_args = ["--stage", "train", "--epochs", str(args.epochs)]
    eval_args = ["--stage", "eval"]

    if args.only:
        ids[args.only] = run_stage(args.only, ["--stage", args.only])
    elif args.all:
        ids["distill"] = run_stage("distill", distill_args)
        ids["scicode_data"] = run_stage("scicode_data", scicode_args)
        ids["code_data"] = run_stage("code_data", code_args)
        ids["train"] = run_stage("train", train_args + [
            "--upstream-distill", ids["distill"],
            "--upstream-scicode", ids["scicode_data"],
            "--upstream-code", ids["code_data"]])
        ids["eval"] = run_stage("eval", eval_args + ["--upstream-train", ids["train"],
                                                     "--upstream-distill", ids["distill"],
                                                     "--upstream-scicode", ids["scicode_data"]])
        ids["convert_gguf"] = run_stage(
            "convert_gguf", ["--stage", "convert_gguf", "--upstream-train", ids["train"]])
    elif args.from_stage:
        if args.from_stage == "code_data":
            ids["code_data"] = run_stage("code_data", code_args)
        if args.from_stage in ("code_data", "train"):
            code_id = ids.get("code_data", args.upstream_code)
            ids["train"] = run_stage("train", train_args + [
                "--upstream-distill", args.upstream_distill,
                "--upstream-scicode", args.upstream_scicode,
                "--upstream-code", code_id,
                "--resume-train", args.resume_train,
                "--epoch-offset", str(args.epoch_offset)])
            ids["eval"] = run_stage("eval", eval_args + ["--upstream-train", ids["train"],
                                                         "--upstream-distill", args.upstream_distill,
                                                         "--upstream-scicode", args.upstream_scicode])
            ids["convert_gguf"] = run_stage(
                "convert_gguf", ["--stage", "convert_gguf", "--upstream-train", ids["train"]])
        elif args.from_stage == "eval":
            ids["eval"] = run_stage("eval", eval_args + ["--upstream-train", args.upstream_train,
                                                         "--upstream-distill", args.upstream_distill,
                                                         "--upstream-scicode", args.upstream_scicode])
            ids["convert_gguf"] = run_stage(
                "convert_gguf", ["--stage", "convert_gguf", "--upstream-train", args.upstream_train])
        else:
            ids["convert_gguf"] = run_stage(
                "convert_gguf", ["--stage", "convert_gguf", "--upstream-train", args.upstream_train])

    print("\n==== clearml-finetuning pipeline summary ====")
    from clearml import Task

    for name, tid in ids.items():
        t = Task.get_task(task_id=tid)
        print(f"{name:8s} {t.status:10s} {tid}  "
              f"https://app.clearml01.nersc.gov/projects/{t.project}/tasks/{tid}/output/log")
    print(f"total wall time: {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
