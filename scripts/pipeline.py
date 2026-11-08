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
sys.path.insert(0, REPO_DIR)
from clearml_finetuning.cml import DEFAULT_OUTPUT_URI  # noqa: E402
QUEUE = os.environ.get("CLEARML_QUEUE", "muller")
DOCKER = os.environ.get("CML_DOCKER", "nvcr.io/nvidia/cuda:13.1.1-runtime-ubuntu24.04")
REPO_URL = os.environ.get("CML_REPO_URL", "https://github.com/tylern4/clearml-finetuning.git")
BRANCH = os.environ.get("CML_REPO_BRANCH", "main")
TERMINAL = {"completed", "failed", "stopped"}

# per-stage venv isolation: vllm and peft stacks must not share a venv
REQUIREMENTS = {
    "distill": ["vllm>=0.11.0", "ninja", "nvidia-cuda-nvrtc==13.0.88",
                "clearml>=2.1", "boto3", "matplotlib"],
    "scicode_data": ["clearml>=2.1", "boto3", "gdown", "h5py", "numpy", "matplotlib"],
    "code_data": ["clearml>=2.1", "boto3", "huggingface_hub", "datasets",
                  "pandas", "pyarrow", "matplotlib"],
    "code_distill": ["vllm>=0.11.0", "ninja", "nvidia-cuda-nvrtc==13.0.88",
                     "clearml>=2.1", "boto3", "matplotlib"],
    "train": ["torch>=2.9.0", "transformers>=4.57", "peft>=0.17", "accelerate>=1.10",
              "bitsandbytes>=0.46", "clearml>=2.1", "boto3", "matplotlib"],
    "eval": ["vllm>=0.11.0", "ninja", "nvidia-cuda-nvrtc==13.0.88", "peft>=0.17",
             "h5py", "numpy", "scipy", "clearml>=2.1", "boto3", "matplotlib",
             "# rebuild venv 2026-11-04: poisoned venvs-cache entry lacked vllm"],
    "convert_gguf": ["torch>=2.9.0", "transformers>=4.57", "peft>=0.17",
                     "gguf>=0.13", "huggingface_hub", "clearml>=2.1", "boto3", "matplotlib"],
}
STAGE_TIMEOUTS = {"distill": 6 * 3600, "scicode_data": 2 * 3600, "code_data": 2 * 3600,
                  "code_distill": 6 * 3600, "train": 12 * 3600,
                  "eval": 8 * 3600, "convert_gguf": 3 * 3600}


def git_head():
    out = subprocess.run(["git", "rev-parse", f"origin/{BRANCH}"],
                         cwd=REPO_DIR, capture_output=True, text=True)
    return out.stdout.strip()


def submit(stage_args: list) -> str:
    """Run the stage runner locally until it self-enqueues; return the task id."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CLEARML_", "TRAINS_", "WF_"))}
    env["WF_OUTPUT_URI"] = os.environ.get("WF_OUTPUT_URI", DEFAULT_OUTPUT_URI)
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


def try_stage(name: str, stage_args: list):
    """Like run_stage but returns (task_id, status) instead of raising."""
    print(f"== stage {name} ==")
    tid = submit(stage_args)
    finalize(tid, name)
    print(f"  task {tid} enqueued to '{QUEUE}'")
    add_tags(tid, [name])
    st = wait_for(tid, STAGE_TIMEOUTS[name])
    print(f"  stage {name}: {st}  ({tid})")
    return tid, st


def add_tags(task_id: str, tags: list) -> None:
    """Best-effort tag merge on a task (keeps existing tags; never raises)."""
    try:
        from clearml import Task
        Task.get_task(task_id=task_id).add_tags(list(tags))
    except Exception as e:
        print(f"  add_tags({task_id}, {tags}) failed: {e}")


def spawn_gguf(train_id: str, epoch: int) -> str:
    """Enqueue a convert_gguf side task and return at once - it runs in
    parallel with the gate eval so a downloadable GGUF exists at every eval
    point for external testing. Returns the task id ("" if the spawn itself
    failed; never raises - the GGUF is a side deliverable, not a gate)."""
    try:
        print("== stage convert_gguf (side task, parallel with eval) ==")
        tid = submit(["--stage", "convert_gguf", "--upstream-train", train_id])
        finalize(tid, "convert_gguf")
        add_tags(tid, ["convert_gguf", f"epoch{epoch}", "gguf"])
        print(f"  task {tid} enqueued to '{QUEUE}'")
        return tid
    except Exception as e:
        print(f"  WARNING: convert_gguf spawn failed (eval continues): {e}")
        return ""


def collect_gguf(gguf_id: str, epoch: int) -> None:
    """Wait for the side GGUF task and tag the outcome. Non-fatal: a failed
    GGUF conversion never gates training."""
    if not gguf_id:
        return
    st = wait_for(gguf_id, STAGE_TIMEOUTS["convert_gguf"])
    print(f"  side gguf: {st}  ({gguf_id})")
    add_tags(gguf_id, ["gguf-ok" if st == "completed" else "gguf-failed"])


def get_scalar(task_id: str, metric: str):
    """Last reported value of a ClearML scalar metric, or None.

    ``metric`` may be dotted "title/series" (e.g. "eval/scicode_main_pass1",
    as reported by run.log({...}) keys containing "/") or a bare series name
    matched against any title.
    """
    from clearml import Task

    try:
        scalars = Task.get_task(task_id=task_id).get_reported_scalars() or {}
        if "/" in metric:
            title_want, series_want = metric.split("/", 1)
        else:
            title_want, series_want = None, metric
        for _title, metrics in scalars.items():
            if title_want and _title != title_want:
                continue
            for m, vals in (metrics or {}).items():
                if m != series_want:
                    continue
                # get_reported_scalars yields {title: {series: {"y": [...]}}}
                ys = vals.get("y") if isinstance(vals, dict) else vals
                if ys:
                    return ys[-1]
    except Exception as e:
        print(f"  get_scalar({metric}) failed: {e}")
    return None


def clearml_api(endpoint: str, body: dict):
    """POST to the ClearML API with creds from the environment (the supervisor
    launch sources clearml-finetuning/.env; nothing is hardcoded here)."""
    import base64
    import json
    import urllib.request

    host = os.environ.get("CLEARML_API_HOST", "")
    key = os.environ.get("CLEARML_API_ACCESS_KEY", "")
    secret = os.environ.get("CLEARML_API_SECRET_KEY", "")
    if not (host and key and secret):
        print("  clearml_api: CLEARML_API_* not set (source .env); skipping")
        return None
    auth = base64.b64encode(f"{key}:{secret}".encode()).decode()
    req = urllib.request.Request(
        host + "/" + endpoint,
        data=json.dumps(body).encode(),
        headers={"Authorization": "Basic " + auth,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def task_artifact_names(task_id: str) -> set:
    """Names of the artifacts a task uploaded (empty set on any error)."""
    from clearml import Task

    try:
        return set(Task.get_task(task_id=task_id).artifacts or {})
    except Exception as e:
        print(f"  task_artifact_names({task_id}) failed: {e}")
        return set()


def mark_partial(task_id: str) -> bool:
    """Mark a walltime-killed train chunk completed + walltime-partial.

    A chunk that trained until the 8h Slurm wall and saved a continuation
    checkpoint did its job for this window; the supervisor resumes from that
    checkpoint, so the task should read completed, not failed. Best-effort:
    returns False (with a warning) if the flip didn't take."""
    try:
        from clearml import Task
        Task.get_task(task_id=task_id).add_tags(["walltime-partial"])
    except Exception as e:
        print(f"  add_tags({task_id}) failed: {e}")
    try:
        r = clearml_api("v2.13/tasks.edit", {
            "task": task_id, "status": "completed", "force": True,
            "status_reason": "8h Slurm walltime; partial chunk, continued by a later chunk",
        })
        if r and r.get("meta", {}).get("result_code") == 200:
            return True
    except Exception as e:
        print(f"  tasks.edit({task_id}) failed: {e}")
    print(f"  WARNING: could not mark {task_id} completed+walltime-partial")
    return False


def run_code_distill(args, ids) -> None:
    """Teacher-solve the code_data problems. Retry once, then abort: the
    teacher-solved mix-in is a hard requirement for this run, and silently
    training without it would waste hours of GPU on the wrong dataset."""
    code_id = ids.get("code_data", args.upstream_code)
    for attempt in (1, 2):
        tid, st = try_stage("code_distill", ["--stage", "code_distill",
                                             "--upstream-code", code_id])
        if st == "completed":
            ids["code_distill"] = tid
            return
        print(f"  WARNING: code_distill attempt {attempt} failed ({st})")
        if attempt == 1:
            print("  retrying code_distill once...")
    raise RuntimeError("code_distill failed twice; the teacher-solved mix-in "
                       "is required — fix the stage and relaunch from "
                       "code_distill (upstream tasks are reused)")


def supervised_train(args, ids) -> None:
    """Keep launching train chunks until a stop criterion is met.

    Chunks resume LoRA from the previous chunk's adapter. Stop when either
    --max-total-epochs is reached or --eval-gate metric >= --eval-gate-min
    after a chunk. A failed chunk is retried (up to --max-train-retries)
    resuming from its latest checkpoint artifact.
    """
    total = args.epoch_offset
    resume = args.resume_train
    best_val = None       # best gate value seen so far (any chunk)
    best_train = ""       # train task id that produced best_val
    best_total = 0        # epoch count at best_val
    if resume:
        st = wait_for(resume, 12 * 3600)
        print(f"resume source {resume}: {st}")
    while True:
        chunk = args.epochs
        if args.max_total_epochs > 0:
            if total >= args.max_total_epochs:
                print(f"reached max total epochs ({total})")
                break
            chunk = min(chunk, args.max_total_epochs - total)
        distill_id = ids.get("distill", args.upstream_distill)
        scicode_id = ids.get("scicode_data", args.upstream_scicode)
        code_id = ids.get("code_data", args.upstream_code)
        cdistill_id = ids.get("code_distill", args.upstream_code_distill)
        ok = False
        attempt = 0
        partials = 0
        while attempt <= args.max_train_retries:
            # rebuild args every attempt: a retry may resume from a different
            # task than the previous attempt (never the dead chunk itself)
            targs = ["--stage", "train", "--epochs", str(chunk),
                     "--upstream-distill", distill_id,
                     "--upstream-scicode", scicode_id,
                     "--upstream-code", code_id,
                     "--upstream-code-distill", cdistill_id,
                     "--resume-train", resume,
                     "--epoch-offset", str(total)]
            if args.extra_data:
                targs += ["--extra-data", args.extra_data]
            if args.train_max_len:
                targs += ["--max-len", str(args.train_max_len)]
            tid, st = try_stage("train", targs)
            arts = task_artifact_names(tid)
            if "student-adapter" in arts:
                # chunk finished its epochs and uploaded the final adapter
                ids["train"] = tid
                resume = tid
                ok = True
                break
            if st == "completed" and "student-adapter-ckpt" in arts:
                # walltime partial: the chunk trained until the 8h Slurm wall
                # and saved a continuation checkpoint (the new SIGTERM handler
                # makes the task read completed natively). Continue the SAME
                # chunk from that checkpoint; this consumes no retry because
                # every continuation makes real forward progress.
                mark_partial(tid)
                print(f"  train chunk partial at walltime ({tid}); continuing "
                      f"it from its checkpoint")
                resume = tid
                partials += 1
                if partials > 8:
                    raise RuntimeError("train chunk hit 8 consecutive walltime "
                                       "partials without completing; giving up "
                                       "rather than looping forever")
                continue
            if "student-adapter-ckpt" in arts:
                print(f"  train chunk failed ({st}); attempt {attempt + 1}/"
                      f"{args.max_train_retries + 1} - will resume from its "
                      f"checkpoint")
                resume = tid
            else:
                # no artifacts at all: a genuine failure. Never resume the dead
                # chunk itself (that crash-loops); fall back to the last good
                # source, or scratch if this was the first chunk.
                print(f"  train chunk failed ({st}) with no checkpoint; attempt "
                      f"{attempt + 1}/{args.max_train_retries + 1} - resuming "
                      f"from last good source {resume or '(scratch)'}")
            attempt += 1
        if not ok:
            raise RuntimeError("train chunk failed after all retries")
        total += chunk
        print(f"total epochs done: {total}")
        add_tags(tid, [f"epoch{total}"])
        if args.eval_gate:
            # external-testing deliverable: convert this chunk's adapter to
            # GGUF while the gate eval runs (separate task, same queue)
            gguf_id = ""
            if not args.no_side_gguf:
                gguf_id = spawn_gguf(ids["train"], total)
            eval_id, st = try_stage("eval", ["--stage", "eval",
                                             "--upstream-train", ids["train"],
                                             "--upstream-distill", distill_id,
                                             "--upstream-scicode", scicode_id])
            ids["eval"] = eval_id
            add_tags(eval_id, [f"epoch{total}", "gate-eval"])
            gate_passed = False
            if st != "completed":
                print(f"  eval after chunk failed ({st}); continuing training")
            else:
                val = get_scalar(eval_id, args.eval_gate)
                print(f"  gate {args.eval_gate} = {val} (need >= {args.eval_gate_min})")
                add_tags(eval_id, ["gate-pass" if val is not None and
                                   val >= args.eval_gate_min else "gate-fail"])
                if val is not None and val >= args.eval_gate_min:
                    print(f"eval gate passed ({args.eval_gate}={val}) - stopping training")
                    gate_passed = True
                elif val is not None and (best_val is None or val > best_val):
                    best_val, best_train, best_total = val, ids["train"], total
                    print(f"  new best gate value: {val} (epoch{total})")
                elif (args.stop_on_eval_decrease and val is not None
                      and best_val is not None and val < best_val):
                    print(f"  gate DECREASED: {val} < best {best_val} "
                          f"(epoch{best_total}) - stopping training")
                    add_tags(eval_id, ["eval-regression"])
                    add_tags(tid, ["eval-regression"])
                    collect_gguf(gguf_id, total)
                    if gguf_id:
                        ids["side_gguf"] = gguf_id
                    if best_train:
                        ids["train"] = best_train
                        print(f"  deliverable rewinds to epoch{best_total} ({best_train})")
                    return
            # collect the side GGUF before moving on (usually already done,
            # it runs alongside eval); failure never gates training
            collect_gguf(gguf_id, total)
            if gguf_id:
                ids["side_gguf"] = gguf_id
            if gate_passed:
                return
        if not args.max_total_epochs and not args.eval_gate:
            break  # single-shot mode (default): one chunk, no gating


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["distill", "scicode_data", "code_data",
                                       "code_distill", "train", "eval", "convert_gguf"])
    ap.add_argument("--from", dest="from_stage",
                    choices=["code_data", "code_distill", "train", "eval",
                             "convert_gguf"])
    ap.add_argument("--upstream-distill", default="")
    ap.add_argument("--upstream-train", default="")
    ap.add_argument("--upstream-scicode", default="")
    ap.add_argument("--upstream-code", default="")
    ap.add_argument("--upstream-code-distill", default="",
                    help="upstream code_distill task id (teacher solutions for the "
                         "code_data problems); consumed by train")
    ap.add_argument("--resume-train", default="",
                    help="previous train task id to resume LoRA from (chunked training)")
    ap.add_argument("--epoch-offset", type=int, default=0,
                    help="epoch seed offset passed to the train stage (for resumed chunks)")
    ap.add_argument("--hf-token", default="",
                    help="HF token for gated datasets in the code_data stage; defaults to "
                         "HF_TOKEN env or $HOME/.hf_token")
    ap.add_argument("--max-total-epochs", type=int, default=0,
                    help="keep launching train chunks (resuming each from the previous) "
                         "until this many total epochs are done; 0 = single chunk")
    ap.add_argument("--eval-gate", default="",
                    help="ClearML scalar metric to gate training on after each chunk, "
                         "e.g. eval/scicode_main_pass1 or eval/pass1")
    ap.add_argument("--eval-gate-min", type=float, default=0.0,
                    help="stop training when --eval-gate >= this value")
    ap.add_argument("--max-train-retries", type=int, default=2,
                    help="retries per train chunk on failure (resumes from checkpoint)")
    ap.add_argument("--extra-data", default="",
                    help="comma-separated local/s3://https urls of extra train data "
                         "(e.g. remote distills) passed through to every train chunk")
    ap.add_argument("--stop-on-eval-decrease", action="store_true",
                    help="stop training when the gate metric drops below the best "
                         "value seen so far (the deliverable rewinds to the best "
                         "chunk, not the regressed one)")
    ap.add_argument("--train-max-len", type=int, default=0,
                    help="token window passed to the train stage (--max-len); "
                         "0 keeps the stage default. Lower this (e.g. 24576) "
                         "when long think rows OOM the 40GB A100s")
    ap.add_argument("--no-side-gguf", action="store_true",
                    help="skip the parallel convert_gguf side task at each eval point")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--epochs", type=int, default=20,
                    help="training epochs passed to the train stage")
    args = ap.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        _tok = os.path.expanduser("~/.hf_token")
        if os.path.exists(_tok):
            with open(_tok) as _f:
                hf_token = _f.read().strip()
    code_args = ["--stage", "code_data"] + (["--hf-token", hf_token] if hf_token else [])

    t0 = time.time()
    ids = {}
    distill_args = ["--stage", "distill"]
    scicode_args = ["--stage", "scicode_data"]
    train_args = ["--stage", "train", "--epochs", str(args.epochs)]
    if args.train_max_len:
        train_args += ["--max-len", str(args.train_max_len)]
    eval_args = ["--stage", "eval"]

    if args.only:
        if args.only == "code_data":
            stage_args = code_args
        elif args.only == "code_distill":
            stage_args = ["--stage", "code_distill", "--upstream-code", args.upstream_code]
        else:
            stage_args = ["--stage", args.only]
        ids[args.only] = run_stage(args.only, stage_args)
    elif args.all:
        ids["distill"] = run_stage("distill", distill_args)
        ids["scicode_data"] = run_stage("scicode_data", scicode_args)
        ids["code_data"] = run_stage("code_data", code_args)
        run_code_distill(args, ids)
        supervised_train(args, ids)
        ids["eval"] = run_stage("eval", eval_args + ["--upstream-train", ids["train"],
                                                     "--upstream-distill", ids["distill"],
                                                     "--upstream-scicode", ids["scicode_data"]])
        if not ids.get("side_gguf"):  # gated campaigns made one per chunk
            ids["convert_gguf"] = run_stage(
                "convert_gguf", ["--stage", "convert_gguf", "--upstream-train", ids["train"]])
    elif args.from_stage:
        if args.from_stage == "code_data":
            ids["code_data"] = run_stage("code_data", code_args)
            run_code_distill(args, ids)
        elif args.from_stage == "code_distill":
            ids["code_data"] = args.upstream_code
            run_code_distill(args, ids)
        if args.from_stage in ("code_data", "code_distill", "train"):
            supervised_train(args, ids)
            ids["eval"] = run_stage("eval", eval_args + ["--upstream-train", ids["train"],
                                                         "--upstream-distill", args.upstream_distill,
                                                         "--upstream-scicode", args.upstream_scicode])
            if not ids.get("side_gguf"):  # gated campaigns made one per chunk
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
