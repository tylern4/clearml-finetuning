"""Stage: evaluate the fine-tuned student.

1. HumanEval pass@1 (greedy, executed in subprocesses with a timeout).
2. Trajectory-format adherence on held-out distilled agent tasks: the student
   must emit valid JSONL agent actions under the scaffold.
Reports metrics to ClearML and uploads sample generations.
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gzip
import urllib.request

from clearml_finetuning import cml, data, prompts  # noqa: E402

_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>.*$", re.DOTALL | re.IGNORECASE)


def strip_think(text):
    """Remove reasoning traces from model output (defensive)."""
    if not isinstance(text, str):
        return text
    text = _THINK_RE.sub("", text)
    return _THINK_OPEN_RE.sub("", text).strip()


def chat(tok, messages):
    """Chat-template with thinking DISABLED.

    Qwen3.5's default template puts the model in thinking mode (the generation
    prompt ends with an open think block). With a 512-token budget that burns
    everything on reasoning and leaves no code, so eval must disable it
    (same as the distill stage).
    """
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def load_humaneval(attempts=3):
    """Fetch HumanEval directly (avoids the datasets/hub version matrix)."""
    urls = [
        "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
        "https://raw.githubusercontent.com/openai/human-eval/main/data/HumanEval.jsonl.gz",
    ]
    last = None
    for url in urls:
        for i in range(attempts):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    blob = gzip.decompress(r.read())
                return [json.loads(l) for l in blob.decode().splitlines() if l.strip()]
            except Exception as e:
                last = e
                print(f"HumanEval download failed ({e}); retry {i + 1}/{attempts}")
    raise last


def main() -> None:
    ap = argparse.ArgumentParser(description="evaluate student")
    ap.add_argument("--student", default=os.environ.get("CML_STUDENT",
                    "Qwen/Qwen3.8-27B"))
    ap.add_argument("--adapter-task", type=str, default="", help="upstream train task id")
    ap.add_argument("--adapter-dir", type=str, default="student_adapter")
    ap.add_argument("--data-task", type=str, default="", help="upstream distill task id (for val rows)")
    ap.add_argument("--data-dir", type=str, default="distilled")
    ap.add_argument("--humaneval-n", type=int, default=0, help="0 = all 164 problems")
    ap.add_argument("--adherence-n", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--scicode-task", type=str, default="", help="upstream scicode_data task id")
    ap.add_argument("--scicode-dir", type=str, default="scicode_test")
    args = ap.parse_args()

    run = cml.get_run_logger("eval", vars(args), task_type="testing")
    cml.prepare_vllm_nvrtc()  # harmless for BF16 students; needed for FP8 checkpoints

    try:
        from vllm import LLM
        from vllm.lora.request import LoRARequest
    except ModuleNotFoundError:
        # Self-heal: Task.init's requirements auto-capture (a freeze of the
        # submitting shell's venv, torch-cpu, no vllm) can win the race against
        # the code-pinned requirements and key a stage venv without vllm.
        # Install the eval stack into the current venv; pip is idempotent, so
        # this is a one-time cost per venv and later runs are no-ops.
        print("[eval] vllm missing in stage venv - installing eval stack", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch"],
                       check=False)
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "vllm>=0.11.0", "ninja", "nvidia-cuda-nvrtc==13.0.88",
                        "peft>=0.17", "h5py", "scipy"], check=True)
        from vllm import LLM
        from vllm.lora.request import LoRARequest

    nproc = int(os.environ.get("NPROC", "4"))
    if args.adapter_task and not os.path.isdir(args.adapter_dir):
        local = cml.download_artifact(args.adapter_task, "student-adapter")
        os.makedirs(os.path.dirname(args.adapter_dir) or ".", exist_ok=True)
        os.rename(local, args.adapter_dir) if not os.path.exists(args.adapter_dir) else None
    use_lora = os.path.isdir(args.adapter_dir) and os.path.exists(
        os.path.join(args.adapter_dir, "adapter_config.json"))

    llm = LLM(
        model=args.student,
        tensor_parallel_size=max(1, min(nproc, 4)),
        max_model_len=32768,
        enable_lora=use_lora,
        max_lora_rank=64 if use_lora else None,
        gpu_memory_utilization=0.90,
    )
    tok = llm.get_tokenizer()
    lora_req = LoRARequest("student", 1, args.adapter_dir) if use_lora else None

    # ---- HumanEval pass@1 (with adapter if present) --------------------------
    problems = load_humaneval()
    problems = problems[:args.humaneval_n] if args.humaneval_n else problems
    prompts_ = [
        chat(tok, [{"role": "user", "content": p["prompt"] +
                    "\n\nComplete the following function. Return only the code."}])
        for p in problems
    ]
    sp = llm.get_default_sampling_params()
    sp.temperature = 0.0
    sp.max_tokens = args.max_tokens
    outs = llm.generate(prompts_, sp, lora_request=lora_req)

    passed = 0
    for p, out in zip(problems, outs):
        code = strip_think(out.outputs[0].text)
        if "```" in code:
            blocks = [b for b in code.split("```") if b.strip()]
            code = blocks[0].split("\n", 1)[-1] if blocks else code
        body = p["prompt"] + "\n" + code
        check = p["test"] + f"\ncheck({p['entry_point']})\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write("import math, os, re, sys, string, itertools\n" + body + "\n" + check)
            path = f.name
        try:
            r = subprocess.run([sys.executable, path], capture_output=True, timeout=15)
            passed += int(r.returncode == 0)
        except subprocess.TimeoutExpired:
            pass
        finally:
            os.unlink(path)
    pass1 = passed / max(1, len(problems))
    print(f"HumanEval pass@1: {passed}/{len(problems)} = {pass1:.3f}")
    if run:
        run.log({"eval/humaneval_passed": passed, "eval/humaneval_n": len(problems),
                 "eval/pass1": pass1}, step=0)

    # ---- trajectory adherence (self-play agent loop) -------------------------
    # The student is trained (build_sft_rows) on multi-turn trajectories: ONE
    # action JSON per assistant turn, the simulated observation as the next
    # user turn. A single-shot completion therefore can never satisfy the old
    # one-shot JSONL check (it correctly emits one action and stops - 0/22 on
    # every eval so far). Instead, replay the loop the training data encodes:
    # generate an action, let the model simulate the environment observation,
    # repeat until finish, then validate the assembled transcript with
    # data.validate_transcript (unchanged).
    if args.data_task and not os.path.isdir(args.data_dir):
        local = cml.download_artifact(args.data_task, "distilled-data")
        os.makedirs(args.data_dir, exist_ok=True)
        import shutil
        for f in os.listdir(local):
            shutil.copy(os.path.join(local, f), args.data_dir)
    val_rows = data.read_jsonl(os.path.join(args.data_dir, "val.jsonl"))
    system_prompt = (val_rows[0]["messages"][0]["content"]
                     if val_rows and "messages" in val_rows[0]
                     else prompts.AGENT_SYSTEM_PROMPT)

    import random

    rng = random.Random(0)
    sample = rng.sample(val_rows, min(args.adherence_n, len(val_rows))) if val_rows else []

    def _first_action(text):
        """First ACTION_KEYS JSON object in a completion, or None."""
        for line in strip_think(text).splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("action") in data.ACTION_KEYS:
                return obj
        return None

    sp_act = llm.get_default_sampling_params()
    sp_act.temperature = 0.0
    sp_act.max_tokens = 400
    sp_obs = llm.get_default_sampling_params()
    sp_obs.temperature = 0.0
    sp_obs.max_tokens = 700

    trajectories = [{
        "row": r,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": f"{r['repo_context']}\n\nTask: {r['task']}"}],
        "lines": [], "n_actions": 0, "done": False, "reason": None,
    } for r in sample]

    # the validator caps transcripts at 8 actions (incl. finish); 9 turns is
    # the hard ceiling - one more turn than allowed so overrun is visible
    for _turn in range(9):
        live = [t for t in trajectories if not t["done"]]
        if not live:
            break
        outs = llm.generate([chat(tok, t["messages"]) for t in live],
                            sp_act, lora_request=lora_req)
        obs_jobs = []
        for t, out in zip(live, outs):
            act = _first_action(out.outputs[0].text)
            if act is None:
                t["done"], t["reason"] = True, "no-valid-action"
                continue
            line = json.dumps(act, ensure_ascii=False)
            t["messages"].append({"role": "assistant", "content": line})
            t["lines"].append(line)
            t["n_actions"] += 1
            if act["action"] == "finish":
                t["done"] = True
            elif t["n_actions"] >= 9:
                t["done"], t["reason"] = True, "too-many-actions"
            else:
                obs_jobs.append(t)
        if not obs_jobs:
            break
        # self-play the environment: the model simulates each observation in
        # the simulator role (the distill teacher did the same when building
        # the training data)
        sim_prompts = [chat(tok, [{"role": "system",
                                   "content": prompts.SIMULATOR_SYSTEM_PROMPT}]
                            + t["messages"][1:]) for t in obs_jobs]
        outs = llm.generate(sim_prompts, sp_obs, lora_request=lora_req)
        for t, out in zip(obs_jobs, outs):
            obs = strip_think(out.outputs[0].text).strip()
            if not obs:
                t["done"], t["reason"] = True, "empty-observation"
                continue
            obs = obs[:1800]
            t["messages"].append({"role": "user",
                                  "content": f"Observation:\n{obs}"})
            t["lines"].append(json.dumps({"observation": obs},
                                         ensure_ascii=False))

    valid, reasons = 0, {}
    n_valid_shown = 0
    for t in trajectories:
        text = "\n".join(t["lines"])
        if t["reason"] is None and data.validate_transcript(text) is not None:
            valid += 1
            if run and n_valid_shown < 2:
                n_valid_shown += 1
                run.report_sample(
                    f"valid self-play trajectory for '{t['row']['title']}':\n{text[:1800]}",
                    step=1, label=f"valid_{t['row']['fingerprint']}")
            continue
        t["reason"] = t["reason"] or ("no-finish" if t["n_actions"] else "no-actions")
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
        if run:
            run.report_sample(
                f"invalid trajectory ({t['reason']}) for '{t['row']['title']}':\n{text[:1800]}",
                step=1, label=f"invalid_{t['row']['fingerprint']}")
    frac = valid / max(1, len(sample))
    print(f"trajectory adherence (self-play): {valid}/{len(sample)} = {frac:.3f} "
          f"failures={reasons}")
    if run:
        run.log({"eval/agent_tasks": len(sample), "eval/valid_trajectories": valid,
                 "eval/format_adherence": frac}, step=1)

    # ---- a couple of end-to-end agent samples as debug samples ---------------
    demo_prompts = [
        "Fix the bug in utils/date.py where Feb 29 crashes the parser.",
        "Add a unit test for the retry helper in client/retry.py.",
    ]
    sp_demo = llm.get_default_sampling_params()
    sp_demo.temperature = 0.0
    sp_demo.max_tokens = 700
    for i, dp in enumerate(demo_prompts):
        ptext = chat(tok, [{"role": "system", "content": system_prompt},
                           {"role": "user", "content": dp}])
        out = strip_think(llm.generate([ptext], sp_demo, lora_request=lora_req)[0].outputs[0].text)
        if run:
            run.report_sample(f"task: {dp}\n\nstudent actions:\n{out[:1200]}",
                              step=2 + i, label=f"demo_{i}")

    # ---- SciCode test split (standard setup: no background, carry over the
    # model's own generated code for previous sub-steps) -----------------------
    if args.scicode_task:
        run_scicode_eval(args, run, llm, tok, lora_req)

    if run:
        run.finish()


def _h5_sparse_matrix(group):
    """Rebuild a scipy sparse matrix from its h5 group (official layout)."""
    import scipy.sparse

    data = group["data"][()]
    shape = tuple(group["shape"][()])
    if "row" in group and "col" in group:
        return scipy.sparse.coo_matrix(
            (data, (group["row"][()], group["col"][()])), shape=shape)
    if "blocksize" in group:
        return scipy.sparse.bsr_matrix(
            (data, group["indices"][()], group["indptr"][()]),
            shape=shape, blocksize=tuple(group["blocksize"][()]))
    return scipy.sparse.csr_matrix(
        (data, group["indices"][()], group["indptr"][()]), shape=shape)


def _h5_datagroup(group):
    """varN group -> list / sparse matrix / dict, per the official harness."""
    import h5py

    for key in group.keys():
        if key == "list":
            return [group[key][k][()] for k in group[key].keys()]
        if key == "sparse_matrix":
            return _h5_sparse_matrix(group[key])
        # otherwise a dict (keys may be float-parseable -> convert to float)
        out = {}
        for k, obj in group.items():
            if isinstance(obj, h5py.Group):
                out[k] = _h5_sparse_matrix(obj["sparse_matrix"])
            elif isinstance(obj[()], bytes):
                out[k] = obj[()].decode("utf-8", errors="strict")
            else:
                try:
                    out[float(k)] = obj[()]
                except ValueError:
                    out[k] = obj[()]
        return out
    return None


def _h5_targets(h5_path):
    """step_id -> list of per-test target payloads. Faithful port of SciCode's
    process_hdf5_to_tuple: single var -> value (decoded if bytes); multiple
    vars -> tuple; nested var groups -> list / sparse matrix / dict."""
    import h5py

    out = {}
    with h5py.File(h5_path, "r") as f:
        for step_id in f.keys():
            group = f[step_id]
            tests = []
            for i in sorted(int(k[4:]) for k in group.keys()):
                tg = group[f"test{i}"]
                keys = list(tg.keys())
                vals = []
                for k in keys:
                    sub = tg[k]
                    if isinstance(sub, h5py.Dataset):
                        v = sub[()]
                        vals.append(v.decode("utf-8", errors="strict")
                                    if isinstance(v, bytes) else v)
                    else:
                        vals.append(_h5_datagroup(sub))
                tests.append(vals[0] if len(vals) == 1 else tuple(vals))
            out[step_id] = tests
    return out


def _extract_code(text):
    """Pull the Python implementation out of a generation."""
    text = strip_think(text)
    if "```" in text:
        blocks = [b for b in text.split("```") if b.strip()]
        if blocks:
            inner = blocks[0].split("\n", 1)[-1] \
                if blocks[0].lstrip().lower().startswith("python") else blocks[0]
            text = inner
    return text.strip()


def _report_eval_plots(run, problems, step_pass, main_pass, n_sub_pass, n_sub):
    """Purpose-built eval figures for the task's Plots tab (latest-wins cards)."""
    if not run:
        return
    try:
        import time

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # headline rates vs the gate
        rows = [("SciCode main (gate)", main_pass / max(1, len(problems))),
                ("SciCode sub-step", (n_sub_pass / n_sub) if n_sub else 0.0)]
        for key, label in (("eval/pass1", "HumanEval"),
                           ("eval/format_adherence", "traj adherence")):
            pts = run._history.get(key) or []
            if pts:
                rows.append((label, pts[-1][1]))
        fig, ax = plt.subplots(figsize=(8, 3.6))
        bars = ax.barh([r[0] for r in rows][::-1],
                       [100 * r[1] for r in rows][::-1], color="#4878cf")
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
        ax.axvline(15, color="crimson", ls="--", lw=1)
        ax.text(15.5, len(rows) - 1.1, "gate 15%", color="crimson", fontsize=8)
        ax.set_xlim(0, 105)
        ax.set_xlabel("pass rate (%)")
        fig.tight_layout()
        run.report_figure("eval_summary", fig)

        # per-problem resolved fraction (where the model struggles)
        frac = []
        for p in problems:
            n = len(p["sub_steps"])
            ok = sum(1 for j in range(n) if step_pass.get((p["problem_id"], j)))
            frac.append((p.get("problem_name") or p["problem_id"], ok / n,
                         ok == n))
        frac.sort(key=lambda t: t[1])
        fig, ax = plt.subplots(figsize=(9, max(6, 0.16 * len(frac) + 1)))
        ax.barh([t[0][:30] for t in frac], [100 * t[1] for t in frac],
                color=["#2ca02c" if t[2] else "#4878cf" for t in frac])
        ax.set_xlim(0, 100)
        ax.set_xlabel("sub-steps passed (%)  ·  green = fully resolved")
        ax.set_title(f"SciCode per-problem ({main_pass}/{len(problems)} resolved)")
        fig.tight_layout()
        run.report_figure("scicode_per_problem", fig, step=1)

        # pass rate by position in the sequential sub-step chain
        by_step = {}
        for p in problems:
            for j in range(len(p["sub_steps"])):
                by_step.setdefault(j, []).append(
                    bool(step_pass.get((p["problem_id"], j))))
        if by_step:
            js = sorted(by_step)
            rates = [100 * sum(by_step[j]) / len(by_step[j]) for j in js]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar([j + 1 for j in js], rates, color="#4878cf")
            for j, r in zip(js, rates):
                ax.text(j + 1, r + 1.5, f"{r:.0f}%\nn={len(by_step[j])}",
                        ha="center", fontsize=8)
            ax.set_ylim(0, 108)
            ax.set_xticks([j + 1 for j in js])
            ax.set_xlabel("sub-step position (sequential generation)")
            ax.set_ylabel("pass rate (%)")
            ax.set_title("SciCode pass rate by sub-step position")
            fig.tight_layout()
            run.report_figure("scicode_by_step", fig, step=2)
    except Exception as ex:
        print(f"[eval] warning: plot reporting failed: {ex}")
    finally:
        # figure conversion/upload trails by seconds; settle before the
        # stage runner exits the process
        try:
            run.task.flush()
            time.sleep(3)
        except Exception:
            pass


def run_scicode_eval(args, run, llm, tok, lora_req):
    import shutil
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    if not os.path.isdir(args.scicode_dir):
        local = cml.download_artifact(args.scicode_task, "scicode-test")
        os.makedirs(args.scicode_dir, exist_ok=True)
        for f in os.listdir(local):
            shutil.copy(os.path.join(local, f), args.scicode_dir)
    test_jsonl = os.path.join(args.scicode_dir, "problems_test.jsonl")
    h5_path = os.path.join(args.scicode_dir, "test_data.h5")
    problems = [json.loads(l) for l in open(test_jsonl) if l.strip()]
    targets = _h5_targets(h5_path)
    print(f"SciCode eval: {len(problems)} test problems, "
          f"{sum(len(p['sub_steps']) for p in problems)} sub-steps")

    sp = llm.get_default_sampling_params()
    sp.temperature = 0.0
    sp.max_tokens = 768

    # generate step-by-step per problem (batch across problems per step index)
    generated = {}  # (problem_id, step_idx) -> code
    max_steps = max(len(p["sub_steps"]) for p in problems)
    for step_idx in range(max_steps):
        batch, keys = [], []
        for p in problems:
            if step_idx >= len(p["sub_steps"]):
                continue
            ss = p["sub_steps"][step_idx]
            prev = "\n\n".join(
                generated[(p["problem_id"], j)] for j in range(step_idx))
            if len(prev) > 12000:  # keep the most recent context, bound prompt size
                prev = "...\n\n" + prev[-12000:]
            user = prompts.SCICODE_USER_TEMPLATE.format(
                problem_description=p["problem_description_main"],
                dependencies=p.get("required_dependencies", ""),
                step_description=ss["step_description_prompt"],
                function_header=ss["function_header"],
            )
            if prev:
                user += f"\n\nSolutions to previous sub-steps:\n{prev}"
            batch.append(chat(tok, [{"role": "system",
                                     "content": prompts.SCICODE_SYSTEM_PROMPT},
                                    {"role": "user", "content": user}]))
            keys.append((p["problem_id"], step_idx))
        outs = llm.generate(batch, sp, lora_request=lora_req)
        for k, out in zip(keys, outs):
            generated[k] = _extract_code(out.outputs[0].text)
        print(f"  generated sub-step column {step_idx + 1}/{max_steps} "
              f"({len(keys)} problems)")

    # execute each sub-step against its h5-backed test cases
    def check_one(item):
        p, step_idx = item
        ss = p["sub_steps"][step_idx]
        step_id = ss["step_number"]
        code = p.get("required_dependencies", "") + "\n"
        for j in range(step_idx + 1):
            code += generated[(p["problem_id"], j)] + "\n"
        tests = ss["test_cases"]
        tvals = targets.get(step_id, [])
        import base64
        import pickle
        blob = base64.b64encode(pickle.dumps(tvals)).decode("ascii")
        script = (code + "\nimport base64, pickle\n"
                  f"targets = pickle.loads(base64.b64decode({blob!r}))\n")
        for i, t in enumerate(tests):
            script += f"\ntarget = targets[{i}]\n" + t + "\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(script)
            path = f.name
        try:
            r = subprocess.run([sys.executable, path], capture_output=True,
                               timeout=120)
            return (p["problem_id"], step_idx, r.returncode == 0,
                    (r.stderr or "").strip().splitlines()[-1][:120]
                    if r.returncode else "")
        except subprocess.TimeoutExpired:
            return (p["problem_id"], step_idx, False, "timeout")
        finally:
            os.unlink(path)

    items = [(p, j) for p in problems for j in range(len(p["sub_steps"]))]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(check_one, items))

    step_pass = {(pid, j): ok for pid, j, ok, _ in results}
    n_sub = len(results)
    n_sub_pass = sum(1 for _, _, ok, _ in results if ok)
    main_pass = sum(
        1 for p in problems
        if all(step_pass.get((p["problem_id"], j), False)
               for j in range(len(p["sub_steps"]))))
    print(f"SciCode sub-step pass@1: {n_sub_pass}/{n_sub} = {n_sub_pass / n_sub:.3f}")
    print(f"SciCode main-problem resolved: {main_pass}/{len(problems)} "
          f"= {main_pass / len(problems):.3f}")
    # Durable result channel: scalar reporting inside the agent can be lost
    # (SDK reporting thread vs vLLM process spawning) and stage stdout is
    # block-buffered through the agent pipe. An artifact uploaded via the
    # current task always survives.
    _metrics = {"scicode_main_pass1": main_pass / len(problems),
                "scicode_sub_pass1": (n_sub_pass / n_sub) if n_sub else 0.0,
                "scicode_main_resolved": main_pass,
                "scicode_main_total": len(problems),
                "scicode_sub_passed": n_sub_pass,
                "scicode_sub_total": n_sub}
    try:
        from clearml import Task as _Task
        _t = _Task.current_task()
        if _t is not None:
            _t.upload_artifact("eval-results", _metrics)
            _fails = [{"problem": pid, "step": j + 1,
                       "error": (err or "")[:400],
                       "generated": (generated.get((pid, j), "") or "")[:1200]}
                      for pid, j, ok, err in [r for r in results if not r[2]][:20]]
            if _fails:
                _t.upload_artifact("scicode-failures", _fails)
                print(f"[eval] uploaded {len(_fails)} failure records "
                      f"as artifact 'scicode-failures'")
            _t.get_logger().report_scalar(title="eval",
                                          series="scicode_main_pass1",
                                          value=_metrics["scicode_main_pass1"],
                                          iteration=3)
    except Exception as _ex:
        print(f"[eval] warning: durable result upload failed: {_ex}")
    _report_eval_plots(run, problems, step_pass, main_pass, n_sub_pass, n_sub)
    if run:
        run.log({"eval/scicode_sub_pass1": n_sub_pass / n_sub,
                 "eval/scicode_sub_passed": n_sub_pass, "eval/scicode_sub_total": n_sub,
                 "eval/scicode_main_resolved": main_pass,
                 "eval/scicode_main_total": len(problems),
                 "eval/scicode_main_pass1": main_pass / len(problems)}, step=3)
        fails = [r for r in results if not r[2]][:2]
        for pid, j, _, err in fails:
            p = next(pp for pp in problems if pp["problem_id"] == pid)
            if run:
                run.report_sample(
                    f"FAILED scicode {pid}.{j + 1} ({p['problem_name']}): {err}\n\n"
                    f"generated:\n{generated[(pid, j)][:1200]}",
                    step=3, label=f"scicode_fail_{pid}_{j}")


if __name__ == "__main__":
    main()
