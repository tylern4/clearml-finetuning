#!/usr/bin/env bash
#SBATCH -J gguf-convert
#SBATCH -A dasrepo
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -o gguf_convert_%j.out

# Convert a fine-tuned LoRA adapter into quantized GGUF files (llama.cpp),
# entirely on a Perlmutter CPU node.
#
# Usage:
#   sbatch convert_to_gguf.sh /path/to/adapter [BASE_MODEL]
#
#   adapter    peft adapter dir (e.g. /pscratch/.../ep6_adapter) or a .zip
#              artifact (student_adapter.zip / ckpt-stepNN.zip both work).
#   BASE_MODEL HF repo to merge the adapter into (default: Qwen/Qwen3.8-27B).
#
# Env overrides:
#   QUANT_TYPES  comma list for llama-quantize   (default: Q4_K_M,Q8_0)
#   KEEP_F16=1   keep the f16 GGUF (~55 GB for a 27B) instead of deleting it
#   OUT_DIR      output dir (default: <script dir>/gguf/<base>-<adapter-tag>)
#   LLAMA_REF    pin a llama.cpp tag (default: latest release)
#
# Outputs land on /pscratch (shared), ready to rsync to a laptop:
#   <OUT_DIR>/<base>-agent-<QUANT>.gguf  +  README.md
#
# Typical runtime on a CPU node: model download ~15m + merge ~20m +
# f16 convert ~20m + two quants ~1h  ->  well inside 04:00:00.

set -euo pipefail

ADAPTER="${1:-}"
BASE_MODEL="${2:-Qwen/Qwen3.8-27B}"
if [[ -z "$ADAPTER" ]]; then
    echo "usage: sbatch convert_to_gguf.sh /path/to/adapter [BASE_MODEL]" >&2
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# prefer the project venv (system python3 is 3.6 and cannot run the merge)
VENV_PY=""
for cand in "$HERE/.venv/bin/python" "$HERE/../.venv/bin/python"; do
    if [[ -x "$cand" ]]; then VENV_PY="$cand"; break; fi
done
if [[ -z "$VENV_PY" ]]; then
    VENV_PY="$(command -v python3)"
    echo "WARNING: project venv not found, falling back to $VENV_PY" >&2
fi

QUANT_TYPES="${QUANT_TYPES:-Q4_K_M,Q8_0}"
BASE_TAG="$(basename "$BASE_MODEL" | tr '[:upper:]' '[:lower:]')"
ADAPTER_TAG="$(basename "$ADAPTER" | sed 's/\.zip$//; s/_adapter$//; s/^adapter$//')"
OUT_DIR="${OUT_DIR:-$HERE/gguf/${BASE_TAG}-${ADAPTER_TAG:-merged}}"
WORK="$HERE/gguf/.work_${SLURM_JOB_ID:-manual}"
LOG() { echo "[gguf $(date +%H:%M:%S)] $*"; }

mkdir -p "$OUT_DIR" "$WORK"

# /pscratch is visible from the CPU nodes; keep the HF cache there so repeat
# runs skip the ~55 GB model download
export HF_HOME="${HF_HOME:-$HERE/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$(nproc)}"

# ---------------------------------------------------------------- adapter ---
SRC="$ADAPTER"
if [[ "$ADAPTER" == *.zip ]]; then
    LOG "extracting adapter zip $ADAPTER"
    SRC="$WORK/adapter_x"
    mkdir -p "$SRC"
    "$VENV_PY" - "$ADAPTER" "$SRC" <<'PYEOF'
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])
PYEOF
fi
# locate the peft adapter: top level of a dir, or a single ckpt-stepNN subdir
# when given a periodic-checkpoint zip
ADAPTER_DIR="$SRC"
if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ]]; then
    ADAPTER_DIR="$(dirname "$(find "$SRC" -name adapter_config.json | head -1)")"
fi
if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ]]; then
    echo "no adapter_config.json under $ADAPTER" >&2
    exit 1
fi
LOG "adapter: $ADAPTER_DIR"

# ------------------------------------------------- python deps + llama.cpp --
"$VENV_PY" -m pip install -q -U gguf jinja2 sentencepiece
# NOTE: never install llama.cpp/requirements.txt - it downgrades the venv's
# torch/transformers below what the Qwen3.8 architecture needs.

# share the llama.cpp checkout between runs (clone + build ~10 min)
LLAMA_DIR="${LLAMA_DIR:-$HERE/gguf/.llama_cpp}"
if [[ ! -f "$LLAMA_DIR/convert_hf_to_gguf.py" ]]; then
    REF="${LLAMA_REF:-}"
    if [[ -z "$REF" ]]; then
        REF="$(curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest \
               | grep -m1 tag_name | cut -d'"' -f4 || true)"
    fi
    [[ -n "$REF" ]] || REF=master
    LOG "cloning llama.cpp @ $REF"
    rm -rf "$LLAMA_DIR"
    git clone -q --depth 1 --branch "$REF" https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR" \
        || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
module load gcc 2>/dev/null || true   # newer gcc when available; else default cc
CMAKE_BIN="$(command -v cmake || true)"
if [[ -z "$CMAKE_BIN" ]]; then
    "$VENV_PY" -m pip install -q cmake
    CMAKE_BIN="$("$VENV_PY" -c 'import cmake, os; print(os.path.join(os.path.dirname(cmake.__file__), "data", "bin", "cmake"))')"
fi
if [[ ! -x "$LLAMA_DIR/build/bin/llama-quantize" ]]; then
    LOG "building llama-quantize (CPU only)"
    "$CMAKE_BIN" -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
        -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF > /dev/null
    "$CMAKE_BIN" --build "$LLAMA_DIR/build" --target llama-quantize -j "${SLURM_CPUS_PER_TASK:-8}"
fi
QUANTIZE_BIN="$LLAMA_DIR/build/bin/llama-quantize"

# ---------------------------------------------- merge LoRA into the base ----
MERGED="$WORK/merged"
if [[ ! -f "$MERGED/config.json" ]]; then
    LOG "merging LoRA into $BASE_MODEL on CPU (needs ~110 GB RAM peak)"
    ADAPTER_DIR="$ADAPTER_DIR" BASE_MODEL="$BASE_MODEL" MERGED="$MERGED" "$VENV_PY" <<'PYEOF'
import os, torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    os.environ["BASE_MODEL"], torch_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(os.environ["BASE_MODEL"])
model = PeftModel.from_pretrained(base, os.environ["ADAPTER_DIR"])
model = model.merge_and_unload()
# keep the fine-tune's tokenizer/chat template if the adapter ships one
try:
    tok = AutoTokenizer.from_pretrained(os.environ["ADAPTER_DIR"])
except Exception:
    pass
model.save_pretrained(os.environ["MERGED"])
tok.save_pretrained(os.environ["MERGED"])
print("merged ->", os.environ["MERGED"])
PYEOF
fi
LOG "merged model ready: $MERGED"

# ------------------------------------------------------ convert + quantize --
F16="$OUT_DIR/${BASE_TAG}-agent-f16.gguf"
if [[ ! -f "$F16" ]]; then
    LOG "converting to f16 GGUF (~55 GB)"
    "$VENV_PY" "$LLAMA_DIR/convert_hf_to_gguf.py" "$MERGED" \
        --outfile "$F16" --outtype f16
fi

IFS=',' read -ra QTYPES <<< "$QUANT_TYPES"
MADE=()
for QT in "${QTYPES[@]}"; do
    QT="$(echo "$QT" | xargs)"   # trim
    [[ -n "$QT" ]] || continue
    OUT="$OUT_DIR/${BASE_TAG}-agent-${QT}.gguf"
    if [[ ! -f "$OUT" ]]; then
        LOG "quantizing -> $QT"
        "$QUANTIZE_BIN" "$F16" "$OUT" "$QT"
    fi
    MADE+=("$OUT")
done
FIRST_GGUF="none"
if [[ "${#MADE[@]}" -gt 0 ]]; then FIRST_GGUF="$(basename "${MADE[0]}")"; fi

# ----------------------------------------------------------------- README ---
cat > "$OUT_DIR/README.md" <<EOF
# $BASE_MODEL fine-tune (GGUF)

LoRA adapter \`$ADAPTER\` merged into \`$BASE_MODEL\`, converted and
quantized with llama.cpp. Chat template: ${ADAPTER_DIR}/chat_template.jinja
(bundled from the fine-tune when present).

\`\`\`bash
# interactive
llama-cli -m $FIRST_GGUF -c 8192

# one-shot
llama-cli -m $FIRST_GGUF -c 8192 -p '<task>' --no-display-prompt
\`\`\`
EOF

if [[ "${KEEP_F16:-0}" != "1" && -f "$F16" ]]; then
    rm -f "$F16"
fi
rm -rf "$WORK"

LOG "done. outputs:"
ls -lh "$OUT_DIR"
LOG "copy to laptop, e.g.:  rsync -avP $OUT_DIR/ laptop:models/${BASE_TAG}-${ADAPTER_TAG:-merged}/"
