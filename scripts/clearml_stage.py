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
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAGES = ("distill", "scicode_data", "code_data", "train", "eval", "convert_gguf")


def main() -> None:
    ap = argparse.ArgumentParser(description="clearml-finetuning stage runner")
    # all args optional: when run remotely the agent re-executes with no argv;
    # actual values come from the connected params
    ap.add_argument("--stage", choices=STAGES, default=None)
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--upstream-distill", type=str, default="")
    ap.add_argument("--upstream-train", type=str, default="")
    ap.add_argument("--upstream-scicode", type=str, default="")
    ap.add_argument("--upstream-code", type=str, default="")
    ap.add_argument("--resume-train", type=str, default="",
                    help="previous train task id to resume LoRA from (chunked training)")
    ap.add_argument("--epoch-offset", type=int, default=0,
                    help="epoch seed offset for resumed chunks")
    ap.add_argument("--hf-token", type=str, default="",
                    help="HF token for gated datasets; passed to the code_data stage "
                         "as HF_TOKEN env (worker \$HOME may not have ~/.hf_token)")
    ap.add_argument("--teacher", type=str, default=os.environ.get(
        "CML_TEACHER", "Qwen/Qwen3.8-27B"))
    ap.add_argument("--student", type=str, default=os.environ.get(
        "CML_STUDENT", "Qwen/Qwen3.5-9B"))
    ap.add_argument("--n-tasks", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--humaneval-n", type=int, default=0)
    ap.add_argument("--student-quant", default=os.environ.get(
        "CML_STUDENT_QUANT", "none"),
        help="none|8bit|4bit quantized base for large students")
    args, _ = ap.parse_known_args()

    from clearml import Task

    task = Task.init(
        project_name=os.environ.get("CML_PROJECT")
        or os.environ.get("CLEARML_PROJECT", "clearml-finetuning"),
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
            "student_quant": args.student_quant,
            "upstream_distill": args.upstream_distill,
            "upstream_train": args.upstream_train,
            "upstream_scicode": args.upstream_scicode,
            "upstream_code": args.upstream_code,
            "resume_train": args.resume_train,
            "epoch_offset": args.epoch_offset,
            "hf_token": args.hf_token,
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

    if args.stage:
        # Pin the task to the pushed repo HEAD + stage requirements + docker
        # BEFORE enqueueing: a fast worker can otherwise grab the queued task
        # mid-finalize and the server rejects the edits (ValueError race).
        from scripts.pipeline import BRANCH, DOCKER, REPO_DIR, REPO_URL, REQUIREMENTS

        commit = subprocess.run(
            ["git", "rev-parse", f"origin/{BRANCH}"],
            cwd=REPO_DIR, capture_output=True, text=True).stdout.strip()
        if not commit:
            raise RuntimeError(f"cannot resolve origin/{BRANCH} - push the repo first")
        task._edit(script={
            "repository": REPO_URL,
            "branch": BRANCH,
            "version_num": commit,
            "working_dir": ".",
            "entry_point": "scripts/clearml_stage.py",
            "requirements": {"pip": REQUIREMENTS[args.stage]},
        })
        task.set_base_docker(docker_image=DOCKER)
        task.flush()

    task.execute_remotely(queue_name=os.environ.get("CLEARML_QUEUE", "muller"))

    stage = params["stage"]
    if stage not in STAGES:
        raise RuntimeError(f"missing/invalid stage parameter: {stage!r}")

    import subprocess

    py = sys.executable
    script = {
        "distill": "scripts/distill_data.py",
        "scicode_data": "scripts/scicode_data.py",
        "code_data": "scripts/code_data.py",
        "train": "scripts/train_student.py",
        "eval": "scripts/eval_agent.py",
        "convert_gguf": "scripts/convert_gguf.py",
    }[stage]

    if stage == "distill":
        cmd = [py, script, "--teacher", str(params["teacher"]),
               "--n-tasks", str(params["n_tasks"])]
    elif stage == "scicode_data":
        cmd = [py, script]
    elif stage == "code_data":
        cmd = [py, script]
    elif stage == "train":
        nproc = int(params.get("nproc") or 4)
        cmd = [py, "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node", str(nproc), script,
               "--student", str(params["student"]),
               "--student-quant", str(params.get("student_quant") or "none"),
               "--data-task", str(params["upstream_distill"]),
               "--scicode-task", str(params["upstream_scicode"]),
               "--code-task", str(params["upstream_code"]),
               "--resume-task", str(params.get("resume_train") or ""),
               "--epoch-offset", str(params.get("epoch_offset") or 0),
               "--epochs", str(params["epochs"]),
               "--lora-r", str(params["lora_r"])]
    elif stage == "eval":
        cmd = [py, script, "--student", str(params["student"]),
               "--adapter-task", str(params["upstream_train"]),
               "--data-task", str(params["upstream_distill"]),
               "--scicode-task", str(params["upstream_scicode"]),
               "--humaneval-n", str(params["humaneval_n"])]
    elif stage == "convert_gguf":
        cmd = [py, script, "--student", str(params["student"]),
               "--adapter-task", str(params["upstream_train"])]

    env = dict(os.environ)
    env["CLEARML_RUN"] = "1"
    if stage == "code_data" and params.get("hf_token"):
        env["HF_TOKEN"] = str(params["hf_token"])
    if stage in ("distill", "eval"):
        # avoid the flashinfer JIT sampler (needs ninja + slow first-use compile)
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    print(f"+ {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"stage '{stage}' script failed with exit code {proc.returncode}")
    print(f"stage '{stage}' complete")


if __name__ == "__main__":
    main()
