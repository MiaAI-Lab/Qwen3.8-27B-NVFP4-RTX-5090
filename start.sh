#!/usr/bin/env bash
#
# start.sh — serve RadixArk/Qwen3.8-27B-NVFP4 with vLLM 0.27.1 on port 8888,
# tuned for a single RTX 5090 (sm_120, 32 GB) that may also drive a desktop.
#
# Profile: full native 262144 (256K) context, TurboQuant 4-bit KV pinned at
# 5.5 GiB, MTP-3 speculative decoding, single session (--max-num-seqs 1).
# Measured on this class of box: ~160 tok/s single-stream, 0/15 garble
# battery fails AFTER the built-in PR #40914 patch (stock 0.27.1 garbles
# 13/15 — see below).
#
# This script:
#   * creates a local .venv and installs stock vLLM 0.27.x once
#   * patches the installed vLLM with the upstream PR #40914 backport
#     (TurboQuant K+1 spec-verify routing — without it, MTP over 4-bit KV
#     captures a context-blind FULL cudagraph at boot and output degenerates
#     into repetition loops / broken tool calls; vLLM issue #40880)
#   * downloads the model weights to MODEL_DIR if missing
#   * starts the server and streams the log until it answers, then returns
#
# Env knobs set for you:
#   * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  (MTP-3 OOMs without it)
#   * VLLM_USE_FLASHINFER_SAMPLER=0  (flashinfer sampler JIT needs nvcc; skip)
#   * adaptive --gpu-memory-utilization from live nvidia-smi (desktop-safe)
#   * CC wrapper giving Triton's JIT shim a Python.h if the OS lacks one
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8888}"
SERVED_MODEL_NAME="qwen38-nvfp4"
MAX_MODEL_LEN=262144                       # native ceiling
KV_CACHE_BYTES=5905580032                  # 5.5 GiB pin: one full 256K req + MTP drafts needs >5.0 GiB
MODEL_REPO="RadixArk/Qwen3.8-27B-NVFP4"
MODEL_DIR="${MODEL_DIR:-$HOME/models/RadixArk/Qwen3.8-27B-NVFP4}"
PINNED_PKGS=( "vllm==0.27.1" "flashinfer-python>=0.6.13" "nvidia-cutlass-dsl>=4.5.2" )

VENV="$SCRIPT_DIR/.venv"
LOG="$SCRIPT_DIR/vllm.log"
PID_FILE="$SCRIPT_DIR/vllm.pid"
CC_WRAP="$HOME/.local/bin/cc-triton"

log() { printf '\033[1;36m[start.sh]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[start.sh] ERROR:\033[0m %b\n' "$*" >&2; exit 1; }

# --- 0. already running? ----------------------------------------------------
if curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  log "a server is already responding on port $PORT — nothing to do."
  exit 0
fi
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  log "server is already starting (pid $(cat "$PID_FILE")) — log: $LOG"
  exit 0
fi

command -v curl >/dev/null || die "curl is required."
command -v python3 >/dev/null || die "python3 is required."
command -v nvidia-smi >/dev/null || die "nvidia-smi is required."

# --- 1. venv + pinned packages (once) ---------------------------------------
if [ ! -x "$VENV/bin/vllm" ]; then
  log "creating venv at $VENV and installing ${PINNED_PKGS[*]} (first run only)…"
  python3 -c "import ensurepip" >/dev/null 2>&1 \
    && python3 -m venv "$VENV" \
    || python3 -m venv --without-pip "$VENV"
  if [ ! -x "$VENV/bin/pip" ]; then
    "$VENV/bin/python" /dev/stdin <<'PY'
import urllib.request; urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", "/tmp/get-pip.py")
import runpy; runpy.run_path("/tmp/get-pip.py", run_name="__main__")
PY
  fi
  "$VENV/bin/pip" install -q -U pip
  "$VENV/bin/pip" install -q "${PINNED_PKGS[@]}" || die "vLLM install failed"
fi
[ -x "$VENV/bin/vllm" ] || die "vllm binary not found in the venv"

# --- 2. MTP garble fix: backport upstream PR #40914 into installed vLLM -----
# Stock 0.27.1 captures the MTP verify step as a context-free first-chunk
# flash_attn FULL cudagraph (dummy batch has seq_len == query_len), so the
# replayed graph never reads the KV cache -> repetition collapse. Route
# uniform K+1 spec-verify batches through the TQ decode kernel instead.
TQ_FILE="$(ls "$VENV"/lib/python3*/site-packages/vllm/v1/attention/backends/turboquant_attn.py 2>/dev/null | head -1)"
[ -f "$TQ_FILE" ] || die "turboquant_attn.py not found in $VENV (vLLM layout changed?)"
if grep -q "spec-verify routing (backport of PR #40914)" "$TQ_FILE"; then
  log "PR #40914 patch already applied — skipping"
else
  log "applying PR #40914 backport (MTP x TurboQuant garble fix)…"
  cp "$TQ_FILE" "$TQ_FILE.pre40914.bak"
  "$VENV/bin/python" - "$TQ_FILE" <<'PY'
import sys
path = sys.argv[1]
src = open(path).read()
ANCHOR = """        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        if not attn_metadata.is_prefill:"""
PATCH = """        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        # --- Spec-decode K+1 spec-verify routing (backport of PR #40914) ---
        # MTP verify produces uniform q_len=K+1 batches with prior cached KV.
        # FULL cudagraph capture feeds a dummy batch with seq_len == q_len, so
        # _prefill_attention would capture the first-chunk flash_attn branch
        # (cu_seqlens_k == cu_seqlens_q) — the replayed graph never reads the
        # KV cache and output degenerates into repetition loops (vllm#40880).
        # Route these batches through the decode kernel with all-GPU synth
        # args instead (same synth_seq_lens trick as the continuation path;
        # no CPU-side per-request branching, so capture/replay stay valid).
        _k1 = attn_metadata.max_query_len
        if (
            attn_metadata.is_prefill
            and num_decodes == 0
            and 1 < _k1 <= 16
            and attn_metadata.max_seq_len > _k1
            and N > 0
            and (N % _k1) == 0
            and attn_metadata.query_start_loc is not None
            and attn_metadata.query_start_loc.shape[0] == N // _k1 + 1
        ):
            _b = N // _k1
            _offs = torch.arange(_k1, device=device, dtype=attn_metadata.seq_lens.dtype)
            _synth_seq_lens = (
                attn_metadata.seq_lens[:_b, None] - _k1 + 1 + _offs[None, :]
            ).reshape(-1)
            _synth_bt = attn_metadata.block_table[:_b].repeat_interleave(_k1, dim=0)
            _synth_meta = TurboQuantMetadata(
                seq_lens=_synth_seq_lens,
                slot_mapping=attn_metadata.slot_mapping,
                block_table=_synth_bt,
                query_start_loc=attn_metadata.query_start_loc,
                num_actual_tokens=N,
                max_query_len=1,
                max_seq_len=attn_metadata.max_seq_len,
                is_prefill=False,
            )
            attn_out = self._decode_attention(
                q, kv_cache, _synth_meta, Pi, centroids, PiT, layer
            )
            if output.ndim == 3:
                output[:N] = attn_out.to(output.dtype)
            else:
                output[:N] = attn_out.reshape(N, -1).to(output.dtype)
            return output

        if not attn_metadata.is_prefill:"""
if src.count(ANCHOR) != 1:
    sys.exit("anchor not found — installed vLLM no longer matches the 0.27.1 "
             "layout; check if PR #40914 has merged upstream before serving "
             "(stock MTP x TurboQuant garbles output)")
open(path, "w").write(src.replace(ANCHOR, PATCH))
print("patched:", path)
PY
fi

# --- 3. model weights --------------------------------------------------------
if [ -f "$MODEL_DIR/config.json" ] && ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
  log "model present at $MODEL_DIR"
else
  log "downloading $MODEL_REPO -> $MODEL_DIR (~22 GiB, resumable)…"
  mkdir -p "$MODEL_DIR"
  "$VENV/bin/python" - "$MODEL_REPO" "$MODEL_DIR" <<'PY' || die "model download failed (re-run resumes)"
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PY
fi

# --- 4. env knobs -------------------------------------------------------------
# MTP-3 on 32 GB is memory-tight; expandable_segments stops the allocator
# stranding 200+ MiB in fragmentation (n=3 warmup OOMs without it).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# flashinfer's top-k/top-p sampler JIT-links with nvcc on first use; vLLM's
# native sampler is the zero-cost fallback on boxes without a CUDA toolkit.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# Adaptive GPU util: 0.9x only works on a headless card. If the 5090 also
# drives a display, the desktop holds ~1-2 GiB and vLLM refuses high utils.
# Size from live nvidia-smi, leave ~0.8 GiB, clamp [0.80, 0.98].
GPU_UTIL="${GPU_UTIL:-$(python3 - <<'PY'
import subprocess
q = lambda k: float(subprocess.check_output(
    ["nvidia-smi", f"--query-gpu=memory.{k}", "--format=csv,noheader,nounits"]).strip())
t, f = q("total"), q("free")
print(f"{max(0.80, min(0.98, round((f - 820.0) / t, 2))):.2f}")
PY
)}"

# Triton's JIT C shim #includes <Python.h>; boxes without python3-dev fail at
# model load. Give Triton headers from a uv-managed Python, no root.
if ! printf '#include <Python.h>\n' | /usr/bin/gcc -E - >/dev/null 2>&1; then
  if [ ! -x "$CC_WRAP" ]; then
    "$VENV/bin/pip" install -q uv >/dev/null 2>&1 || true
    [ -x "$VENV/bin/uv" ] && "$VENV/bin/uv" python install 3.14 --install-dir "$HOME/.uv-pythons" >/dev/null 2>&1 || true
    mkdir -p "$HOME/.local/bin"
    cat > "$CC_WRAP" <<'EOF'
#!/usr/bin/env bash
for d in "$HOME"/.uv-pythons/*/include/python3.1*/; do
  [ -f "${d}Python.h" ] && exec /usr/bin/gcc -I"${d}" "$@"
done
exec /usr/bin/gcc "$@"
EOF
    chmod +x "$CC_WRAP"
  fi
  export CC="$CC_WRAP"
fi

# --- 5. launch the server -----------------------------------------------------
: > "$LOG"
SERVE_ARGS=(
  serve "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0 --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --kv-cache-dtype turboquant_4bit_nc
  --kv-cache-memory-bytes "$KV_CACHE_BYTES"
  --max-num-seqs 1                 # MTP + concurrency crashes this KV path
  --max-num-batched-tokens 512     # bound long-prefill activation peaks
  --gpu-memory-utilization "$GPU_UTIL"
  --attention-config.flash_attn_version=2
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
)

log "starting vLLM on port $PORT (gpu_util=$GPU_UTIL) …"
nohup "$VENV/bin/vllm" "${SERVE_ARGS[@]}" >>"$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
disown "$SERVER_PID" 2>/dev/null || true   # keep running after this script exits

# --- 6. stream the log until the server answers -------------------------------
tail -n +1 -F "$LOG" &
TAIL_PID=$!
trap 'kill "$TAIL_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 300); do
  if curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$TAIL_PID" 2>/dev/null || true
    echo "" >&2
    die "vLLM exited during startup — last log lines:\n$(tail -n 60 "$LOG")"
  fi
  sleep 2
done
curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
  || die "server did not become healthy in 10 min — see $LOG"

# --- 7. warm the large-prefill path (cold-serve first-big-prompt fix) --------
# A tiny "hello" does NOT compile the large-prefill path; fire one ~26K-token
# request so the first real client prompt doesn't stall or come back stunned.
kill "$TAIL_PID" 2>/dev/null || true
log "warming the large-prefill path (~26K-token prefill)…"
"$VENV/bin/python" - "$PORT" <<'PY' || log "WARN warmup failed (serve may still be compiling)"
import json, sys, urllib.request
port = sys.argv[1]
prompt = ("Unified memory bandwidth bounds decode throughput on edge accelerators today. " * 2200) + "\nReply with: OK"
body = json.dumps({"model": "qwen38-nvfp4", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 8, "temperature": 0,
                   "chat_template_kwargs": {"enable_thinking": False}}).encode()
urllib.request.urlopen(urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
    headers={"Content-Type": "application/json"}), timeout=300).read()
print("warmup ok")
PY

trap - EXIT
echo ""
log "Qwen3.8-27B-NVFP4 is serving"
log " base URL : http://0.0.0.0:$PORT/v1 (OpenAI-compatible API)"
log " model    : $SERVED_MODEL_NAME"
log " profile  : 256K context, turboquant_4bit_nc KV (5.5 GiB), MTP-3, single-session"
log " log file : $LOG"
log " stop     : kill \$(cat $PID_FILE); pkill -f VLLM::EngineCore"
echo ""
exit 0
