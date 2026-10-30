"""Per-stage ClearML task runner for the distill/train/eval pipeline.

Mirrors the nanochat pattern: Task.init + params + output_uri, register mode
for the controller, execute_remotely for self-enqueue, artifact handoff
between stages, subprocess dispatch of the stage scripts.

Usage (from the repo root):
    python -m scripts.clearml_stage --stage distill
    python -m scripts.clearml_stage --stage train --upstream-distill <task_id>
    python -m scripts.clearml_stage --stage eval --upstream-train <id> --upstream-distill <id>
Append --register to only create the task.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAGES = ("distill", "train", "eval")


def main() -> None:
    ap = argparse.ArgumentParser(description="clearml-finetuning stage runner")
    # all args optional: when run remotely the agent re-executes with no argv;
    # actual values come from the connected params
    ap.add_argument("--stage", choices=STAGES, default=None)
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--upstream-distill", type=str, default="")
    ap.add_argument("--upstream-train", type=str, default="")
    ap.add_argument("--teacher", type=str, default=os.environ.get(
        "CML_TEACHER", "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"))
    ap.add_argument("--student", type=str, default=os.environ.get(
        "CML_STUDENT", "Qwen/Qwen2.5-Coder-7B-Instruct"))
    ap.add_argument("--n-tasks", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--humaneval-n", type=int, default=0)
    args, _ = ap.parse_known_args()

    from clearml import Task

    task = Task.init(
        project_name=os.environ.get("CLEARML_PROJECT", "clearml-finetuning"),
        task_name=f"finetune-{args.stage or 'draft'}",
        task_type=Task.TaskTypes.training,
        auto_connect_frameworks=False,
        reuse_last_task_id=False,
    )
    params = task.connect(
        {
            "stage": args.stage,
            "teacher": args.teacher,
            "student": args.student,
            "n_tasks": args.n_tasks,
            "epochs": args.epochs,
            "lora_r": args.lora_r,
            "humaneval_n": args.humaneval_n,
            "upstream_distill": args.upstream_distill,
            "upstream_train": args.upstream_train,
            "nproc": int(os.environ.get("NPROC", "4")),
        },
        name="stage",
    )
    output_uri = os.environ.get("WF_OUTPUT_URI")
    if output_uri:
        task.output_uri = output_uri
        try:
            task.get_logger().set_default_upload_destination(output_uri)
        except Exception:
            pass

    if args.register:
        print(f"register stage task: id={task.id}")
        task.close()
        return

    task.execute_remotely(queue_name=os.environ.get("CLEARML_QUEUE", "muller"))

    stage = params["stage"]
    if stage not in STAGES:
        raise RuntimeError(f"missing/invalid stage parameter: {stage!r}")

    import subprocess

    py = sys.executable
    script = {
        "distill": "scripts/distill_data.py",
        "train": "scripts/train_student.py",
        "eval": "scripts/eval_agent.py",
    }[stage]

    cmd = [py, script,
           "--teacher", str(params["teacher"]),
           "--student", str(params["student"])]
    if stage == "distill":
        cmd += ["--n-tasks", str(params["n_tasks"])]
    elif stage == "train":
        cmd += ["--data-task", str(params["upstream_distill"]),
                "--epochs", str(params["epochs"]),
                "--lora-r", str(params["lora_r"])]
    elif stage == "eval":
        cmd += ["--adapter-task", str(params["upstream_train"]),
                "--data-task", str(params["upstream_distill"]),
                "--humaneval-n", str(params["humaneval_n"])]

    env = dict(os.environ)
    env["CLEARML_RUN"] = "1"
    print(f"+ {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"stage '{stage}' script failed with exit code {proc.returncode}")
    print(f"stage '{stage}' complete")


if __name__ == "__main__":
    main()
