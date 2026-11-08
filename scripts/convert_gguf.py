"""Stage: convert the fine-tuned student to GGUF for llama.cpp / Ollama.

Steps:
  1. merge the LoRA adapter into the base student model (CPU, bf16)
  2. fetch a pinned llama.cpp release
  3. convert to GGUF (f16)
  4. build llama-quantize and produce laptop-friendly quants (Q4_K_M, Q8_0)
  5. upload the GGUF files (+ a run-me README) as a ClearML artifact
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clearml_finetuning import cml  # noqa: E402


def sh(cmd, env=None, cwd=None):
    print(f"+ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, env=env, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="convert student to GGUF")
    ap.add_argument("--student", default=os.environ.get("CML_STUDENT",
                    "Qwen/Qwen3.5-9B"))
    ap.add_argument("--adapter-task", type=str, default="")
    ap.add_argument("--adapter-dir", type=str, default="student_adapter")
    ap.add_argument("--llama-ref", type=str, default="",
                    help="llama.cpp tag/commit to pin (empty = latest release)")
    ap.add_argument("--quant-types", type=str, default="Q4_K_M,Q8_0")
    ap.add_argument("--out-dir", type=str, default="gguf")
    ap.add_argument("--work-dir", type=str, default="llamacpp_work")
    args = ap.parse_args()

    run = cml.get_run_logger("convert_gguf", vars(args), task_type="testing")

    # ---- 1. restore adapter artifact ----------------------------------------
    if args.adapter_task and not os.path.isdir(args.adapter_dir):
        local = cml.download_artifact(args.adapter_task, "student-adapter")
        os.makedirs(args.adapter_dir, exist_ok=True)
        for f in os.listdir(local):
            src = os.path.join(local, f)
            # older train packaging nested ckpt dirs / agent archive-cache
            # staging dirs inside the zip; the merge only needs the flat
            # top-level adapter files
            if os.path.isdir(src):
                continue
            shutil.copy(src, os.path.join(args.adapter_dir, f))

    # ---- 2. merge LoRA into the base model (CPU) -----------------------------
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("merging LoRA adapter into base model (CPU)...")
    model = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.student)
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model = model.merge_and_unload()

    merged_dir = os.path.join(args.work_dir, "merged")
    model.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)
    del model
    print(f"merged model saved to {merged_dir}")

    # ---- 3. fetch llama.cpp at a pinned ref ----------------------------------
    ref = args.llama_ref
    if not ref:
        with urllib.request.urlopen(
                "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
                timeout=30) as r:
            ref = json.loads(r.read())["tag_name"]
        print(f"pinning llama.cpp latest release: {ref}")
    llama_dir = os.path.join(args.work_dir, "llama.cpp")
    if not os.path.isdir(llama_dir):
        sh(["git", "clone", "--depth", "1", "--branch", ref,
            "https://github.com/ggml-org/llama.cpp.git", llama_dir])

    # ---- 4. converter deps + build llama-quantize ----------------------------
    # NOTE: do NOT install llama.cpp/requirements.txt here - it pins torch and
    # an older transformers, downgrading the venv below what the Qwen3.5
    # architecture needs. The converter only needs gguf (+ what's installed).
    sh([sys.executable, "-m", "pip", "install", "-q", "gguf", "jinja2", "sentencepiece"])
    sh(["apt-get", "update", "-qq"], env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"})
    sh(["apt-get", "install", "-y", "-qq", "cmake", "build-essential"],
       env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"})
    sh(["cmake", "-S", llama_dir, "-B", os.path.join(llama_dir, "build"),
        "-DGGML_CUDA=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=OFF"])
    sh(["cmake", "--build", os.path.join(llama_dir, "build"),
        "--target", "llama-quantize", "-j", str(os.cpu_count() or 8)])
    quantize_bin = os.path.join(llama_dir, "build", "bin", "llama-quantize")
    assert os.path.exists(quantize_bin), "llama-quantize build failed"

    # ---- 5. convert f16 then quantize ----------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    base_name = args.student.split("/")[-1].lower()
    f16 = os.path.join(args.out_dir, f"{base_name}-agent-f16.gguf")
    sh([sys.executable, os.path.join(llama_dir, "convert_hf_to_gguf.py"),
        merged_dir, "--outfile", f16, "--outtype", "f16"])

    quants = []
    for qtype in [q.strip() for q in args.quant_types.split(",") if q.strip()]:
        out_path = os.path.join(args.out_dir, f"{base_name}-agent-{qtype}.gguf")
        sh([quantize_bin, f16, out_path, qtype])
        quants.append(out_path)
    if not os.environ.get("CML_KEEP_F16"):
        os.unlink(f16)  # f16 is an intermediate; quants are the deliverables

    # ---- 6. a small run-me README next to the files --------------------------
    readme = os.path.join(args.out_dir, "README.md")
    with open(readme, "w") as f:
        q0 = args.quant_types.split(",")[0].strip()
        f.write(
            f"# {args.student} coding-agent GGUF\n\n"
            f"LoRA fine-tune merged into the base model, converted and quantized "
            f"with llama.cpp @ {ref}.\n\n"
            "## Interactive chat (llama.cpp)\n\n"
            "```bash\n"
            f"llama-cli -m {base_name}-agent-{q0}.gguf -c 8192\n"
            "```\n\n"
            "## One-shot\n\n"
            "```bash\n"
            f"llama-cli -m {base_name}-agent-{q0}.gguf -c 8192 \\\n"
            "    -p '<task>' --no-display-prompt\n"
            "```\n\n"
            "The model was fine-tuned with a fixed agent scaffold (JSONL tool "
            "actions). For scaffold-formatted responses, include the system "
            "prompt from clearml_finetuning/prompts.py (AGENT_SYSTEM_PROMPT) as "
            "the system message, e.g. via llama-server:\n\n"
            "```bash\n"
            f"llama-server -m {base_name}-agent-{q0}.gguf -c 8192\n"
            "curl http://localhost:8080/v1/chat/completions \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            "  -d '{\"messages\":[{\"role\":\"system\",\"content\":\"<AGENT_SYSTEM_PROMPT>\"},"
            "{\"role\":\"user\",\"content\":\"<task>\"}]}'\n"
            "```\n\n"
            "GPU offload: add `-ngl 99` (CUDA/Metal); Q4_K_M runs fine CPU-only.\n"
        )

    if run:
        from pathlib import Path

        # Upload the whole output directory as ONE artifact so every GGUF
        # (plus README) is preserved under artifacts/gguf/. Uploading each
        # file separately under the same name "gguf" would overwrite the
        # artifact pointer each time, leaving only the last file registered.
        run.task.upload_artifact("gguf", Path(args.out_dir), wait_on_upload=True)
        run.finish()


if __name__ == "__main__":
    main()
