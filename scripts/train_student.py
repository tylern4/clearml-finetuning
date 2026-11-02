"""Stage: LoRA SFT of the student model on distilled agent trajectories.

Minimal DDP training loop (nanochat style): bf16 base model, peft LoRA,
assistant-only loss masking via the tokenizer's chat template, cosine LR
schedule, gradient accumulation. Saves the adapter as a ClearML artifact.
"""

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from clearml_finetuning import cml, data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA SFT training")
    ap.add_argument("--student", default=os.environ.get("CML_STUDENT",
                    "Qwen/Qwen3.5-9B"))
    ap.add_argument("--data-task", type=str, default="", help="upstream distill task id")
    ap.add_argument("--data-dir", type=str, default="distilled", help="local data dir if already present")
    ap.add_argument("--scicode-task", type=str, default="", help="upstream scicode_data task id")
    ap.add_argument("--scicode-dir", type=str, default="scicode", help="local scicode dir if present")
    ap.add_argument("--student-quant", default=os.environ.get(
        "CML_STUDENT_QUANT", "none"),
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
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--out-dir", default="student_adapter")
    args = ap.parse_args()

    is_ddp = "RANK" in os.environ
    if is_ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    master = rank == 0
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
                    shutil.copy(os.path.join(local, f), dest_dir)
            else:
                shutil.copy(local, target_file or os.path.join(dest_dir, os.path.basename(local)))
            open(ready, "w").close()
        else:
            deadline = time.time() + 3600
            while not os.path.exists(ready):
                if time.time() > deadline:
                    raise RuntimeError(f"rank{rank}: timed out waiting for {ready}")
                time.sleep(5)

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
        sft_rows += [r["messages"] for r in scicode_rows]
        n_scicode = len(scicode_rows)
    if master:
        print(f"loaded {len(sft_rows)} training rows "
              f"({len(sft_rows) - n_scicode} agent + {n_scicode} scicode)")

    # tokenize with assistant-only loss mask; fall back to masking the
    # assistant suffix if the chat template lacks generation markers
    def tokenize(msgs):
        try:
            out = tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False,
                return_dict=True, return_assistant_tokens_mask=True,
            )
            ids = out["input_ids"]
            mask = out["assistant_masks"]
        except Exception:
            prompt_ids = tok.apply_chat_template(
                msgs[:-1], tokenize=True, add_generation_prompt=True)
            full_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
            ids = full_ids
            mask = [0] * len(prompt_ids) + [1] * max(0, len(full_ids) - len(prompt_ids))
        labels = [t if m == 1 else -100 for t, m in zip(ids, mask)]
        return ids, labels

    examples = []
    for msgs in sft_rows:
        ids, labels = tokenize(msgs)
        if len(ids) > args.max_len:
            ids, labels = ids[:args.max_len], labels[:args.max_len]
        examples.append((ids, labels))
    if not examples:
        raise RuntimeError(
            f"no trainable rows after tokenization - check the distill stage "
            f"output (data dir: {args.data_dir})")
    if master:
        lens = sorted(len(e[0]) for e in examples)
        print(f"tokenized: {len(examples)} rows, len p50={lens[len(lens)//2]} p95={lens[int(len(lens)*.95)]} max={lens[-1]}")

    # shard across ranks
    shard = examples[rank::world]

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
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
        load_kwargs.pop("torch_dtype", None)
        if torch.cuda.is_available():
            load_kwargs["device_map"] = {"": f"cuda:{local}"}
        if master:
            print(f"loading student base in {quant} (bnb); LoRA stays bf16")
    model = AutoModelForCausalLM.from_pretrained(args.student, **load_kwargs)
    model.config.use_cache = False
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
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

    steps_per_epoch = math.ceil(math.ceil(len(shard) / args.device_batch_size) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if master:
        print(f"steps/epoch={steps_per_epoch} total_steps={total_steps}")

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    run_logger = run.task.get_logger() if run else None
    step, t0, tokens_seen = 0, time.time(), 0
    accum_loss = 0.0
    model.train()
    stop = False
    for epoch in range(args.epochs):
        rng = torch.Generator().manual_seed(1234 + epoch + rank)
        order = torch.randperm(len(shard), generator=rng).tolist()
        micro_batches = [shard[i:i + args.device_batch_size] for i in range(0, len(shard), args.device_batch_size)]
        opt.zero_grad(set_to_none=True)
        for mi, mb in enumerate(micro_batches):
            input_ids, labels, attn = collate(mb)
            input_ids, labels, attn = (input_ids.to(device), labels.to(device), attn.to(device))
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()
            tokens_seen += int(attn.sum().item())
            if (mi + 1) % args.grad_accum == 0 or mi == len(micro_batches) - 1:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
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
                accum_loss, tokens_seen = 0.0, 0
                step += 1
        if is_ddp:
            dist.barrier()

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
