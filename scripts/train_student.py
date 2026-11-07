"""Stage: LoRA SFT of the student model on distilled agent trajectories.

Minimal DDP training loop (nanochat style): bf16 base model, peft LoRA,
assistant-only loss masking via the tokenizer's chat template, cosine LR
schedule, gradient accumulation. Saves the adapter as a ClearML artifact.
"""

import argparse
import json
import math
import os
import re
import signal
import sys
import time
from datetime import timedelta

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from clearml_finetuning import cml, data  # noqa: E402


# Slurm sends SIGTERM at walltime (SIGKILL follows after KillDelay). The
# training loop only ACTS on this flag at optimizer boundaries, so a
# checkpoint save is never interrupted mid-write.
_stop = {"flag": False}


def _on_sigterm(signum, frame):
    _stop["flag"] = True


def ckpt_global_step(resume_dir):
    """Optimizer step recorded in a resumed periodic checkpoint, or 0.

    Checkpoints write resume_meta.json at save time so a resumed job can
    fast-forward the data schedule instead of replaying already-trained
    steps (which would hit the Slurm wall cap at the same point forever).
    Final adapters never carry a root-level resume_meta.json (any ckpt-*
    dirs inside them are subdirectories), so a missing meta means 0.
    """
    meta = os.path.join(resume_dir, "resume_meta.json")
    if os.path.exists(meta):
        try:
            with open(meta) as f:
                return int(json.load(f).get("global_step", 0))
        except Exception:
            pass
    return 0


def ckpt_step_from_artifact(task_id):
    """Legacy fallback: recover the checkpoint step from the artifact file
    name (student-adapter-ckpt/ckpt-stepNNN.zip) of a pre-meta checkpoint."""
    try:
        from clearml import Task

        art = (Task.get_task(task_id=task_id).artifacts or {}) \
            .get("student-adapter-ckpt")
        # SDK Artifact exposes .url (older builds: .uri)
        url = getattr(art, "url", None) or getattr(art, "uri", "") or ""
        m = re.search(r"ckpt-step(\d+)", url)
        return int(m.group(1)) if m else 0
    except Exception:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA SFT training")
    ap.add_argument("--student", default=os.environ.get("CML_STUDENT",
                    "Qwen/Qwen3.8-27B"))
    ap.add_argument("--data-task", type=str, default="", help="upstream distill task id")
    ap.add_argument("--data-dir", type=str, default="distilled", help="local data dir if already present")
    ap.add_argument("--scicode-task", type=str, default="", help="upstream scicode_data task id")
    ap.add_argument("--scicode-dir", type=str, default="scicode", help="local scicode dir if present")
    ap.add_argument("--code-task", type=str, default="", help="upstream code_data task id")
    ap.add_argument("--code-dir", type=str, default="code", help="local code-data dir if present")
    ap.add_argument("--code-distill-task", type=str, default="",
                    help="upstream code_distill task id (teacher solutions for the "
                         "code_data problems)")
    ap.add_argument("--code-distill-dir", type=str, default="code_distilled",
                    help="local code-distilled dir if present")
    ap.add_argument("--extra-data", type=str, default="",
                    help="comma-separated local paths, s3:// or https(s):// urls of "
                         "extra mix-in JSONLs in {'messages': [...]} row format (e.g. "
                         "outputs of scripts/make_remote_distill.py); one entry per "
                         "model, mixed in like the code-distill rows")
    ap.add_argument("--extra-dir", type=str, default="extra_data",
                    help="local dir for fetched --extra-data files")
    ap.add_argument("--resume-task", type=str, default="",
                    help="upstream train task id whose student-adapter artifact to resume LoRA from")
    ap.add_argument("--resume-dir", type=str, default="resume_adapter",
                    help="local dir holding the adapter to resume from")
    ap.add_argument("--save-every", type=int, default=40,
                    help="checkpoint + upload adapter every N optimizer steps (0=off); "
                         "survives an 8h-cap kill so a later job can resume")
    ap.add_argument("--epoch-offset", type=int, default=0,
                    help="added to the epoch seed when resuming (chunked multi-job training)")
    ap.add_argument("--student-quant", default=os.environ.get(
        "CML_STUDENT_QUANT", "4bit"),
        help="none|8bit|4bit: quantize the frozen base so a large student fits "
             "a 40GB GPU under DDP (LoRA adapters stay bf16)")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device-batch-size", type=int, default=1,
                    help="9B bf16 + LoRA on 40GB A100s: keep at 1")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=16384,
                    help="token window; over-window rows are tail-truncated. "
                         "Reasoning rows (think-wrapped traces) need the raise: "
                         "at 2048 the answer tail is cut and the row is dropped "
                         "as unsupervised")
    ap.add_argument("--out-dir", default="student_adapter")
    args = ap.parse_args()

    is_ddp = "RANK" in os.environ
    if is_ddp:
        # 27B 4-bit fp32 + reference linear-attn kernels make steps slow
        # (~2-10 min); the default 600s NCCL watchdog aborts on rank skew.
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    master = rank == 0
    # exit cleanly on the Slurm walltime SIGTERM (see the boundary stop-check)
    signal.signal(signal.SIGTERM, _on_sigterm)
    run = cml.get_run_logger("train", vars(args)) if master else None
    torch.cuda.set_device(local) if torch.cuda.is_available() else None
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.student)

    # ---- data: from upstream artifact or local dir ---------------------------
    # All ranks share one filesystem: let rank 0 download/copy each upstream
    # artifact exactly once and drop a sentinel file; other ranks poll for the
    # sentinel so they never read a half-copied JSONL (truncated-line crash).
    import shutil

    def fetch_upstream(task_id, name, dest_dir, ready_name, target_file=None):
        ready = os.path.join(dest_dir, ready_name)
        if os.path.exists(ready):
            return
        if master:
            local = cml.download_artifact(task_id, name)
            os.makedirs(dest_dir, exist_ok=True)
            if os.path.isdir(local):
                for f in os.listdir(local):
                    src = os.path.join(local, f)
                    dst = os.path.join(dest_dir, f)
                    if os.path.isdir(src):
                        # final adapter artifacts embed their periodic
                        # ckpt-stepNNN subdirs; copy them recursively
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy(src, dst)
            else:
                shutil.copy(local, target_file or os.path.join(dest_dir, os.path.basename(local)))
            open(ready, "w").close()
        else:
            deadline = time.time() + 3600
            while not os.path.exists(ready):
                if time.time() > deadline:
                    raise RuntimeError(f"rank{rank}: timed out waiting for {ready}")
                time.sleep(5)

    def fetch_remote(url, dest_dir, ready_name):
        """Fetch a remote JSONL (s3://, https(s)://, or local path) once per
        node; same sentinel pattern as fetch_upstream. Returns the local path."""
        import hashlib as _h

        ready = os.path.join(dest_dir, ready_name)
        out = os.path.join(
            dest_dir, _h.sha1(url.encode()).hexdigest()[:16] + ".jsonl")
        if os.path.exists(ready):
            return out
        if master:
            os.makedirs(dest_dir, exist_ok=True)
            if os.path.exists(url):
                shutil.copy(url, out)
            else:
                from clearml.storage import StorageManager

                local = StorageManager.get_local_copy(remote_url=url)
                if not local:
                    raise RuntimeError(f"could not fetch extra data: {url}")
                if local.endswith(".zip"):
                    import zipfile

                    zdir = local[:-4] + "_x"
                    if not os.path.isdir(zdir):
                        with zipfile.ZipFile(local) as z:
                            z.extractall(zdir)
                    jsonls = [os.path.join(zdir, f) for f in os.listdir(zdir)
                              if f.endswith(".jsonl")]
                    if len(jsonls) != 1:
                        raise RuntimeError(f"{url}: expected one .jsonl in the "
                                           f"zip, got {jsonls}")
                    local = jsonls[0]
                if os.path.isdir(local):
                    raise RuntimeError(f"extra data {url} unpacked to a directory; "
                                       f"pass a direct .jsonl url")
                shutil.copy(local, out)
            open(ready, "w").close()
        else:
            deadline = time.time() + 3600
            while not os.path.exists(ready):
                if time.time() > deadline:
                    raise RuntimeError(f"rank{rank}: timed out waiting for {ready}")
                time.sleep(5)
        return out

    if args.data_task:
        fetch_upstream(args.data_task, "distilled-data", args.data_dir, ".distilled.ready")

    from clearml_finetuning import prompts

    rows = data.read_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    sft_rows = [r["messages"] for r in data.build_sft_rows(rows, prompts.AGENT_SYSTEM_PROMPT)]

    # ---- optional SciCode mix-in (multi-task: agent scaffold + science code) --
    n_scicode = 0
    scicode_train = os.path.join(args.scicode_dir, "scicode_train.jsonl")
    if args.scicode_task and not os.path.exists(scicode_train):
        fetch_upstream(args.scicode_task, "scicode-data", args.scicode_dir,
                       ".scicode.ready", target_file=scicode_train)
    if os.path.exists(scicode_train):
        scicode_rows = data.read_jsonl(scicode_train)
        sft_rows += [data.normalize_messages(r["messages"]) for r in scicode_rows]
        n_scicode = len(scicode_rows)

    # ---- optional extra code-domain mix-in (bigcodebench / terminal-bench /
    #      SWE-chat / cyber-security rows from the code_data stage) ------------
    n_code = 0
    code_train = os.path.join(args.code_dir, "code_train.jsonl")
    if args.code_task and not os.path.exists(code_train):
        fetch_upstream(args.code_task, "code-data", args.code_dir,
                       ".code.ready", target_file=code_train)
    if os.path.exists(code_train):
        code_rows = data.read_jsonl(code_train)
        sft_rows += [data.normalize_messages(r["messages"]) for r in code_rows]
        n_code = len(code_rows)

    # ---- optional teacher-solved mix-in (code_distill stage: the teacher's own
    #      answers to the code_data problems, across all four HF sources) ------
    n_cdist = 0
    cdist_train = os.path.join(args.code_distill_dir, "code_distilled.jsonl")
    if args.code_distill_task and not os.path.exists(cdist_train):
        fetch_upstream(args.code_distill_task, "code-distilled-data",
                       args.code_distill_dir, ".cdist.ready", target_file=cdist_train)
    if os.path.exists(cdist_train):
        cdist_rows = data.read_jsonl(cdist_train)
        sft_rows += [data.normalize_messages(r["messages"]) for r in cdist_rows]
        n_cdist = len(cdist_rows)

    # ---- optional extra mix-ins (e.g. remote-model distill sets produced by
    #      scripts/make_remote_distill.py; one JSONL url per model) ------------
    for xi, xurl in enumerate([u.strip() for u in args.extra_data.split(",") if u.strip()]):
        xpath = fetch_remote(xurl, args.extra_dir, f".extra{xi}.ready")
        extra_rows = data.read_jsonl(xpath)
        sft_rows += [data.normalize_messages(r["messages"]) for r in extra_rows]
        if master:
            print(f"extra data [{xi}] {len(extra_rows)} rows from {xurl}")

    # ---- optional resume from a previous train job's adapter ----------------
    resume_dir = ""
    ff_step = 0
    if args.resume_task:
        adapter_marker = os.path.join(args.resume_dir, "adapter_config.json")
        resumed_ckpt = False
        if not os.path.exists(adapter_marker):
            try:
                fetch_upstream(args.resume_task, "student-adapter", args.resume_dir,
                               ".resume.ready")
            except RuntimeError:
                # job was killed (e.g. 8h cap) before the final save: fall back
                # to the latest periodic checkpoint artifact
                if master:
                    print(f"no final adapter on {args.resume_task}; "
                          f"resuming from latest student-adapter-ckpt")
                fetch_upstream(args.resume_task, "student-adapter-ckpt",
                               args.resume_dir, ".resume.ready")
                resumed_ckpt = True
        if os.path.exists(adapter_marker):
            resume_dir = args.resume_dir
            ff_step = ckpt_global_step(resume_dir)
            if resumed_ckpt and ff_step == 0:
                ff_step = ckpt_step_from_artifact(args.resume_task)
    # ff_step must be identical on every rank: only the master rank can take
    # the RuntimeError fallback above (non-master ranks never raise from
    # fetch_upstream -- they just wait for the .resume.ready sentinel), so a
    # legacy checkpoint resume leaves rank 0 with ff_step=200 while the other
    # ranks keep 0. Rank 0 then skips straight to the epoch-end barrier while
    # the other ranks train into the void -- mismatched collectives deadlock
    # NCCL until the 3600s watchdog kills all ranks (seen on chunk 15cfdeaa).
    if world > 1:
        ff_t = torch.tensor([int(ff_step)], dtype=torch.long, device=device)
        dist.broadcast(ff_t, src=0)
        ff_step = int(ff_t.item())
    if master:
        print(f"resume adapter: {resume_dir or 'none (fresh LoRA)'}"
              + (f" (fast-forward to step {ff_step})" if ff_step else ""))
    if master:
        print(f"loaded {len(sft_rows)} training rows "
              f"({len(sft_rows) - n_scicode - n_code - n_cdist} agent + {n_scicode} scicode "
              f"+ {n_code} code-data + {n_cdist} code-teacher)")

    # tokenize with assistant-only loss mask; fall back to masking the
    # assistant suffix if the chat template lacks generation markers
    def tokenize(msgs):
        ids, mask = None, None
        try:
            out = tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False,
                return_dict=True, return_assistant_tokens_mask=True,
            )
            ids = out["input_ids"]
            mask = out["assistant_masks"]
        except Exception:
            pass
        if not ids or not any(mask):
            # The chat template lacks {% generation %} markers, so
            # return_assistant_tokens_mask silently returns an all-zero mask
            # (which would make the loss NaN via 0/0). Fall back to masking
            # everything before the final assistant turn.
            prompt_ids = tok.apply_chat_template(
                msgs[:-1], tokenize=True, add_generation_prompt=True,
                return_dict=True)["input_ids"]
            full_ids = tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False,
                return_dict=True)["input_ids"]
            ids = full_ids
            mask = [0] * len(prompt_ids) + [1] * max(0, len(full_ids) - len(prompt_ids))
        labels = [t if m == 1 else -100 for t, m in zip(ids, mask)]
        return ids, labels

    examples = []
    n_empty = 0
    n_bad = 0
    for msgs in sft_rows:
        try:
            ids, labels = tokenize(msgs)
        except Exception as e:
            # e.g. jinja2 TemplateError: No user query found in messages
            n_bad += 1
            continue
        if len(ids) > args.max_len:
            ids, labels = ids[:args.max_len], labels[:args.max_len]
        if all(l == -100 for l in labels):
            # no supervised tokens -> HF loss would be NaN (0/0); drop the row
            n_empty += 1
            continue
        examples.append((ids, labels))
    if master and n_empty:
        print(f"dropped {n_empty} rows with zero supervised tokens")
    if master and n_bad:
        print(f"dropped {n_bad} rows that failed tokenization")
    if not examples:
        raise RuntimeError(
            f"no trainable rows after tokenization - check the distill stage "
            f"output (data dir: {args.data_dir})")
    if master:
        lens = sorted(len(e[0]) for e in examples)
        print(f"tokenized: {len(examples)} rows, len p50={lens[len(lens)//2]} p95={lens[int(len(lens)*.95)]} max={lens[-1]}")

    # Shard micro-batches (not rows) across ranks, padding to a multiple of
    # world size so every rank executes the SAME number of forward/backward
    # (and therefore DDP allreduce) calls — a one-micro-batch count mismatch
    # deadlocks the process group at the end-of-epoch barrier.
    all_mb = [examples[i:i + args.device_batch_size]
              for i in range(0, len(examples), args.device_batch_size)]
    while len(all_mb) % world:
        all_mb.append(all_mb[-1])
    shard = all_mb[rank::world]

    # ---- model ---------------------------------------------------------------
    # DDP keeps a full base-model replica per rank, so a >~20B bf16 student
    # (e.g. Qwen3.8-27B = ~54GB) overflows a 40GB A100. CML_STUDENT_QUANT=8bit
    # (or 4bit) loads the frozen base quantized via bitsandbytes; only the LoRA
    # adapters stay bf16 and train.
    quant = (args.student_quant or "none").lower()
    load_kwargs = {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if quant in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig

        if quant == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            # fp32 compute: the gated-delta-rule (linear attention) scan in
            # bf16 is numerically unstable and produced NaN loss in the first
            # steps; fp32 dequant+matmul keeps it stable (base stays 4-bit).
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_use_double_quant=True)
        load_kwargs.pop("torch_dtype", None)
        if torch.cuda.is_available():
            load_kwargs["device_map"] = {"": f"cuda:{local}"}
        if master:
            print(f"loading student base in {quant} (bnb); LoRA stays bf16")
    model = AutoModelForCausalLM.from_pretrained(args.student, **load_kwargs)
    model.config.use_cache = False
    if quant == "none":
        # hybrid (linear-attention) models run the gated-delta-rule scan in the
        # model dtype; bf16 makes it numerically unstable -> upcast just those
        # modules to fp32 (cheap: they are a fraction of the params)
        n_up = 0
        for _name, mod in model.named_modules():
            cn = type(mod).__name__.lower()
            if "gateddelta" in cn or "linearattention" in cn or "gated_delta" in cn:
                mod.to(torch.float32)
                n_up += 1
        if master and n_up:
            print(f"upcast {n_up} linear-attention modules to fp32")
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    if resume_dir:
        # continue training the previous job's LoRA weights on the fresh base;
        # is_trainable=True is required — from_pretrained defaults to an
        # inference-only load (requires_grad=False on every adapter param,
        # which DDP then refuses to wrap)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, resume_dir, is_trainable=True)
        if master:
            print(f"resumed LoRA weights from {resume_dir}")
    else:
        try:
            model = get_peft_model(model, lora_cfg)
        except ValueError:
            # newer architectures may rename projections; attach LoRA to all
            # linear layers except the head instead
            if master:
                print("standard LoRA targets not found; falling back to all-linear")
            lora_cfg.target_modules = "all-linear"
            model = get_peft_model(model, lora_cfg)
    if master:
        model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.to(device)
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local], output_device=local)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def collate(batch):
        maxlen = max(len(ids) for ids, _ in batch)
        input_ids, labels, attn = [], [], []
        for ids, labels_ in batch:
            pad = maxlen - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            labels.append(labels_ + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))

    steps_per_epoch = math.ceil(len(shard) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if master:
        print(f"steps/epoch={steps_per_epoch} total_steps={total_steps}")

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    def save_ckpt(step_now):
        """Master-side checkpoint at an optimizer boundary: adapter + schedule
        metadata, uploaded as the student-adapter-ckpt artifact (the
        supervisor's continuation source when the 8h wall kills the job)."""
        ckpt = os.path.join(args.out_dir, f"ckpt-step{step_now}")
        os.makedirs(ckpt, exist_ok=True)
        (model.module if hasattr(model, "module") else model).save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        # tells a resumed job how far to fast-forward its data
        # schedule (see ckpt_global_step)
        with open(os.path.join(ckpt, "resume_meta.json"), "w") as mf:
            json.dump({"global_step": step_now,
                       "epoch_offset": args.epoch_offset,
                       "epochs": args.epochs}, mf)
        if run:
            from pathlib import Path
            run.task.upload_artifact(
                "student-adapter-ckpt", Path(ckpt), wait_on_upload=True)
        print(f"saved checkpoint at step {step_now} -> {ckpt}", flush=True)

    run_logger = run.task.get_logger() if run else None
    step, t0, tokens_seen = 0, time.time(), 0
    accum_loss = 0.0
    model.train()
    stop = False
    if ff_step and master:
        print(f"fast-forward: replaying the data schedule through step {ff_step} "
              f"(already trained in the resumed checkpoint) before training",
              flush=True)
    for epoch in range(args.epochs):
        rng = torch.Generator().manual_seed(1234 + epoch + args.epoch_offset)
        order = torch.randperm(len(all_mb), generator=rng).tolist()
        # identical permutation on every rank, then each rank takes its slice
        # so micro-batch counts stay equal across ranks
        micro_batches = [all_mb[i] for i in order][rank::world]
        opt.zero_grad(set_to_none=True)
        for mi, mb in enumerate(micro_batches):
            boundary = (mi + 1) % args.grad_accum == 0 or mi == len(micro_batches) - 1
            if step < ff_step:
                # fast-forward: this job resumes a killed chunk's checkpoint;
                # the data order is deterministic (same seed + epoch_offset),
                # so replaying the schedule without forward/backward/optimizer
                # work keeps the remaining steps aligned with the killed run
                if _stop["flag"]:
                    # Slurm walltime during fast-forward: no compute has run
                    # since the resumed-from checkpoint, so that checkpoint is
                    # still current -- exit 0 WITHOUT saving (a fresh adapter
                    # here would claim "done" while having trained nothing).
                    # Collective-free: every rank got the same SIGTERM and
                    # leaves on its own; the epoch barrier below re-syncs us.
                    stop = True
                    break
                if boundary:
                    step += 1
                    if master and step % 20 == 0:
                        print(f"fast-forward {step}/{ff_step}", flush=True)
                continue
            input_ids, labels, attn = collate(mb)
            input_ids, labels, attn = (input_ids.to(device), labels.to(device), attn.to(device))
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / args.grad_accum
            if not torch.isfinite(loss):
                # non-finite loss would poison the weights via NaN grads; skip
                # this micro-batch entirely
                if master:
                    print(f"step {step} micro {mi}: non-finite loss "
                          f"({loss.item()}), skipping", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            accum_loss += loss.item()
            tokens_seen += int(attn.sum().item())
            if boundary:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                total_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if not torch.isfinite(total_norm):
                    if master:
                        print(f"step {step}: non-finite grad norm, skipping "
                              f"optimizer step", flush=True)
                    opt.zero_grad(set_to_none=True)
                else:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                if master:
                    dt = time.time() - t0
                    t0 = time.time()
                    run_logger and run.log({
                        "step": step,
                        "train/loss": accum_loss,
                        "train/lr": lr_at(step),
                        "train/tok_per_sec": tokens_seen / max(dt, 1e-6),
                        "train/progress_pct": 100 * step / max(1, total_steps),
                    }, step=None)
                    # mid-run plot refresh, same cadence as checkpoints: the
                    # UI Plots tab shows a slider across these snapshots
                    # (run is the ClearMLRun shim; run_logger is the raw
                    # clearml Logger and has no report_progress_plot)
                    if run and args.save_every and step % args.save_every == 0:
                        run.report_progress_plot(step=step)
                accum_loss, tokens_seen = 0.0, 0
                step += 1
                # periodic checkpoint: saved locally + uploaded as an artifact so
                # it survives an 8h-cap kill; a later job resumes via --resume-task
                saved_here = False
                if args.save_every and step % args.save_every == 0 and step > ff_step:
                    if master:
                        save_ckpt(step)
                    saved_here = True
                    if is_ddp:
                        dist.barrier()
                # walltime stop-check, agreed by ALL ranks: Slurm SIGTERMs every
                # rank together, and the all_reduce turns that into one shared
                # decision so no rank trains on while another saves (NCCL hang)
                if is_ddp:
                    stop_t = torch.tensor([1 if _stop["flag"] else 0],
                                          device=device, dtype=torch.int32)
                    dist.all_reduce(stop_t, op=dist.ReduceOp.MAX)
                    hit_stop = bool(stop_t.item())
                else:
                    hit_stop = _stop["flag"]
                if hit_stop:
                    if master and step > ff_step and not saved_here:
                        save_ckpt(step)
                    if is_ddp:
                        dist.barrier()
                    if master:
                        print(f"walltime SIGTERM at step {step}: saved "
                              f"continuation checkpoint, exiting cleanly",
                              flush=True)
                    if is_ddp:
                        dist.destroy_process_group()
                    if run:
                        run.finish()
                    return
        if is_ddp:
            dist.barrier()
        if stop:
            break

    if stop:
        # walltime hit during fast-forward (see the ff-region stop check):
        # nothing new was trained, so finalize without saving or uploading
        if master:
            print("walltime SIGTERM during fast-forward: the resumed-from "
                  "checkpoint is current; exiting without saving", flush=True)
        if is_ddp:
            dist.destroy_process_group()
        if run:
            run.finish()
        return

    if master:
        os.makedirs(args.out_dir, exist_ok=True)
        (model.module if hasattr(model, "module") else model).save_pretrained(args.out_dir)
        tok.save_pretrained(args.out_dir)
        if run:
            from pathlib import Path
            run.task.upload_artifact("student-adapter", Path(args.out_dir), wait_on_upload=True)
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()
    if run:
        run.finish()


if __name__ == "__main__":
    main()
