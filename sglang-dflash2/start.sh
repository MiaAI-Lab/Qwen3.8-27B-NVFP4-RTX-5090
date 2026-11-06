#!/usr/bin/env bash
#
# sglang-dflash2/start.sh — serve RadixArk/Qwen3.8-27B-NVFP4 with SGLang 0.5.23 and the
# z-lab DFlash2 block-diffusion drafter on port 8888, on a single RTX 5090 (sm_120, 32 GB)
# that may also drive a desktop. Same weights as ../start.sh (vLLM); a different engine.
#
# Profile: 48K context, fp8 KV with SGLang's radix (prefix) cache, DFlash2 speculative
# decoding (block 8, fp8 draft weights), bf16 GDN state, single session.
# Measured on one 5090 (README.md here): greedy decode 226-249 tok/s vs 118 for ../start.sh,
# turn-2 TTFT ~0.1 s on a repeated 14K-token prefix.
#
# This script:
#   * bootstraps uv and a uv-managed Python 3.12 venv, installs the pinned lock (once)
#   * installs prebuilt FlashInfer kernels (flashinfer-jit-cache, cu130) so sm_120 needs no JIT
#   * downloads the target (shared with ../start.sh) and the DFlash2 drafter at pinned revisions
#   * starts the server, streams the log until it answers, then warms the large-prefill path
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8888}"
HOST="${HOST:-0.0.0.0}"
API_KEY="${API_KEY:-}"                     # optional; empty = no auth, like ../start.sh
SERVED_MODEL_NAME="qwen38-nvfp4"
MODEL_REPO="RadixArk/Qwen3.8-27B-NVFP4"
MODEL_REV="319f741cce68d7914884900c138a1fbb70a42f30"
MODEL_DIR="${MODEL_DIR:-$HOME/models/RadixArk/Qwen3.8-27B-NVFP4}"
DRAFT_REPO="z-lab/Qwen3.8-27B-DFlash2"     # mirror of incoai/Qwen3.8-27B-DFlash2
DRAFT_REV="50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
DRAFT_DIR="${DRAFT_DIR:-$HOME/models/z-lab/Qwen3.8-27B-DFlash2}"

# Memory: mem-fraction-static is a fraction of the memory that is FREE at start, so on a card
# that drives a desktop the pools would change size with the desktop. --max-total-tokens pins
# the KV pool instead, so every start takes the same VRAM. Headless: see README "Headless".
CONTEXT_LENGTH="${CONTEXT_LENGTH:-49152}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-53248}"   # empty = let mem-fraction-static size the pool
MEM_FRACTION="${MEM_FRACTION:-0.92}"

VENV="$SCRIPT_DIR/.venv"
BOOT="$SCRIPT_DIR/.bootstrap"
LOG="$SCRIPT_DIR/sglang.log"
PID_FILE="$SCRIPT_DIR/sglang.pid"
JIT_CACHE_PKG="flashinfer-jit-cache==0.7.2"
JIT_CACHE_INDEX="https://flashinfer.ai/whl/cu130"

log() { printf '\033[1;36m[sglang-dflash2]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[sglang-dflash2] ERROR:\033[0m %b\n' "$*" >&2; exit 1; }
AUTH=(); [ -n "$API_KEY" ] && AUTH=(-H "Authorization: Bearer $API_KEY")
healthy() { curl -sf --max-time 3 ${AUTH[@]+"${AUTH[@]}"} "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; }

# --- 0. already running? ----------------------------------------------------
if healthy; then
  log "a server is already responding on port $PORT — nothing to do."
  exit 0
fi
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  log "server is already starting (pid $(cat "$PID_FILE")) — log: $LOG"
  exit 0
fi

command -v curl >/dev/null || die "curl is required."
command -v python3 >/dev/null || die "python3 is required (only to bootstrap uv)."
command -v nvidia-smi >/dev/null || die "nvidia-smi is required."

# --- 1. venv + pinned lock (once) --------------------------------------------
if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import sglang" >/dev/null 2>&1; then
  log "creating $VENV (uv-managed Python 3.12) and installing requirements.lock.txt (first run only)…"
  if ! command -v uv >/dev/null; then
    [ -x "$BOOT/bin/uv" ] || { python3 -m venv "$BOOT" && "$BOOT/bin/pip" install -q uv; } \
      || die "could not bootstrap uv (python3-venv missing?)"
    export PATH="$BOOT/bin:$PATH"
  fi
  export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$SCRIPT_DIR/.uv-python}"
  uv venv --python 3.12 --managed-python "$VENV"
  uv pip install --python "$VENV/bin/python" --prerelease=allow -r requirements.lock.txt || die "lock install failed"
  uv pip install --python "$VENV/bin/python" "$JIT_CACHE_PKG" --index-url "$JIT_CACHE_INDEX" \
    || log "WARN flashinfer-jit-cache not installed; FlashInfer will JIT sm_120 kernels on first use (needs nvcc)"
fi
"$VENV/bin/python" -c "import sglang; assert sglang.__version__ == '0.5.23', sglang.__version__" \
  || die "venv does not hold sglang 0.5.23 — delete $VENV and re-run"

# --- 2. weights ---------------------------------------------------------------
fetch() {  # repo rev dir
  if [ -f "$3/config.json" ] && ls "$3"/*.safetensors >/dev/null 2>&1; then
    log "present: $3"
    return
  fi
  log "downloading $1@${2:0:8} -> $3 (resumable)…"
  mkdir -p "$3"
  "$VENV/bin/python" - "$1" "$2" "$3" <<'PY' || die "download of $1 failed (re-run resumes)"
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])
PY
}
fetch "$MODEL_REPO" "$MODEL_REV" "$MODEL_DIR"     # ~22 GiB, shared with ../start.sh
fetch "$DRAFT_REPO" "$DRAFT_REV" "$DRAFT_DIR"     # ~3.6 GiB

# --- 3. env knobs ---------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Inductor compiles with one worker per CPU by default; on a small VM (we measured in an
# 8 GB WSL2 VM) the parallel workers compete with the engine for RAM at boot.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
# The measured runs had the CUDA 13.0 toolkit on PATH; use it when present.
if [ -x /usr/local/cuda-13.0/bin/nvcc ]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}" PATH="/usr/local/cuda-13.0/bin:$PATH"
elif ! command -v nvcc >/dev/null; then
  log "note: no nvcc on PATH (measured with CUDA 13.0 nvcc present; untested without it)"
fi

# --- 4. launch the server -----------------------------------------------------
# A leftover engine from a killed run would hold most of the VRAM.
pgrep -f 'sglang::' >/dev/null && die "a stale sglang:: process is running — run ./stop.sh first"
: > "$LOG"
POOL_ARGS=(); [ -n "$MAX_TOTAL_TOKENS" ] && POOL_ARGS=(--max-total-tokens "$MAX_TOTAL_TOKENS")
KEY_ARGS=(); [ -n "$API_KEY" ] && KEY_ARGS=(--api-key "$API_KEY")
SERVE_ARGS=(
  -m sglang.launch_server
  --model-path "$MODEL_DIR" --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST" --port "$PORT"
  --tp-size 1 --max-running-requests 1
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-metrics
  --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer       # flashinfer on sm_120 (trtllm_mha is sm_100-only)
  --context-length "$CONTEXT_LENGTH" ${POOL_ARGS[@]+"${POOL_ARGS[@]}"}
  --chunked-prefill-size 1024                                     # 2048-token chunk activations do not fit next to the drafter
  --mamba-radix-cache-strategy extra_buffer_lazy                  # 4 GDN state slots per request instead of 5
  --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 8             # 78.4 MB per slot instead of 153.9 MB (fp32)
  --mem-fraction-static "$MEM_FRACTION"
  --speculative-algorithm DFLASH --speculative-draft-model-path "$DRAFT_DIR"
  --speculative-dflash-block-size 8                               # the drafter's trained block (7 drafts + 1)
  --speculative-draft-window-size 8192
  --speculative-draft-model-quantization fp8                      # drafter 2.3 GB instead of 4.5 GB
  --language-only --mm-feature-transport cpu                      # text-only: no vision tower on the GPU
  ${KEY_ARGS[@]+"${KEY_ARGS[@]}"}
)

log "starting SGLang on port $PORT (ctx=$CONTEXT_LENGTH pool=${MAX_TOTAL_TOKENS:-auto} mem=$MEM_FRACTION) …"
nohup "$VENV/bin/python" "${SERVE_ARGS[@]}" >>"$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
disown "$SERVER_PID" 2>/dev/null || true

# --- 5. stream the log until the server answers -------------------------------
tail -n +1 -F "$LOG" &
TAIL_PID=$!
trap 'kill "$TAIL_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 450); do
  healthy && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$TAIL_PID" 2>/dev/null || true
    echo "" >&2
    die "SGLang exited during startup — last log lines:\n$(tail -n 60 "$LOG")"
  fi
  sleep 2
done
healthy || die "server did not become healthy in 15 min — see $LOG"

# --- 6. warm the large-prefill path (same ~26K-token request as ../start.sh) ---
kill "$TAIL_PID" 2>/dev/null || true
log "warming the large-prefill path (~26K-token prefill)…"
"$VENV/bin/python" - "$PORT" "$API_KEY" <<'PY' || log "WARN warmup failed (serve may still be compiling)"
import json, sys, urllib.request
port, key = sys.argv[1], sys.argv[2]
prompt = ("Unified memory bandwidth bounds decode throughput on edge accelerators today. " * 2200) + "\nReply with: OK"
body = json.dumps({"model": "qwen38-nvfp4", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 8, "temperature": 0,
                   "chat_template_kwargs": {"enable_thinking": False}}).encode()
hdr = {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}
urllib.request.urlopen(urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers=hdr), timeout=300).read()
print("warmup ok")
PY

trap - EXIT
echo ""
log "Qwen3.8-27B-NVFP4 is serving (SGLang 0.5.23 + DFlash2)"
log " base URL : http://$HOST:$PORT/v1 (OpenAI-compatible API)"
log " model    : $SERVED_MODEL_NAME"
log " profile  : ${CONTEXT_LENGTH} ctx, fp8 KV + radix cache, DFlash2 block 8, single-session"
log " log file : $LOG"
log " stop     : ./stop.sh"
echo ""
exit 0
