# Qwen3.8-27B-NVFP4 on a Single RTX 5090 (vLLM)

One script to serve **RadixArk/Qwen3.8-27B-NVFP4** with [vLLM](https://github.com/vllm-project/vllm) 0.27.x on a **single NVIDIA RTX 5090 (32 GB, sm_120)** — full native **256K context**, **TurboQuant 4-bit KV cache** pinned at 5.5 GiB, and **MTP-3 speculative decoding**, exposed as an OpenAI-compatible API.

Works fine on a desktop GPU that still drives a display (adaptive `--gpu-memory-utilization` from live `nvidia-smi`).

## Measured

On this class of box (RTX 5090, 32 GB):

- **~160 tok/s** single-stream generation (MTP-3)
- **Full 262,144-token context** resident (KV: 4-bit TurboQuant, 5.5 GiB)
- **0/15** garble-battery fails **with the built-in PR #40914 patch** (stock 0.27.1 garbles 13/15 — see [Why the patch](#why-the-40914-patch))

## Requirements

| | |
|---|---|
| GPU | NVIDIA RTX 5090, 32 GB (`sm_120`) |
| Drivers | recent enough for `sm_120` (vLLM 0.27.x) |
| System tools | `curl`, `python3`, `nvidia-smi`, `gcc` |
| Disk | ~22 GiB for model weights + ~8 GiB for the venv |
| RAM | 32 GB+ comfortable; model is NVFP4 so VRAM does the heavy lifting |

No CUDA toolkit or `python3-dev` required — the script works around both (see [Notes](#notes)).

## Quick start

```bash
./start.sh
```

First run: creates `.venv`, installs `vllm==0.27.1` + `flashinfer-python` + `nvidia-cutlass-dsl`, applies the MTP×TurboQuant fix, downloads the model (~22 GiB, resumable), starts the server on port **8888**, and exits once it's healthy and warmed up. Subsequent runs are near-instant if the server is already up.

When you see:

```
[start.sh] Qwen3.8-27B-NVFP4 is serving
 base URL : http://0.0.0.0:8888/v1 (OpenAI-compatible API)
 model    : qwen38-nvfp4
```

you're done.

### Try it

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen38-nvfp4",
    "messages": [{"role": "user", "content": "Say OK"}],
    "max_tokens": 8,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Or any OpenAI SDK client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8888/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="qwen38-nvfp4",
    messages=[{"role": "user", "content": "Say OK"}],
    max_tokens=8,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(resp.choices[0].message.content)
```

## What `start.sh` does

1. **Idempotent start** — if the server already answers on the port (or a start is in flight), exits immediately.
2. **venv + pinned deps** (once): `vllm==0.27.1`, `flashinfer-python>=0.6.13`, `nvidia-cutlass-dsl>=4.5.2`.
3. **Patches installed vLLM** with a backport of upstream [PR #40914](#why-the-40914-patch) (idempotent; keeps a `.pre40914.bak`).
4. **Downloads model** `RadixArk/Qwen3.8-27B-NVFP4` → `~/models/RadixArk/Qwen3.8-27B-NVFP4` (resumable).
5. **Env knobs**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (MTP-3 OOMs without it), `VLLM_USE_FLASHINFER_SAMPLER=0` (sampler JIT needs `nvcc`; native sampler is the zero-cost fallback), adaptive `--gpu-memory-utilization` (leaves ~0.8 GiB for a desktop, clamped to [0.80, 0.98]).
6. **Launches vLLM** (V1 engine) with:
   - `--max-model-len 262144` (native ceiling)
   - `--kv-cache-dtype turboquant_4bit_nc` + `--kv-cache-memory-bytes 5905580032` (5.5 GiB — one full 256K request + MTP drafts needs >5.0 GiB)
   - `--max-num-seqs 1` (MTP + concurrency crashes this KV path), `--max-num-batched-tokens 512` (bounds long-prefill activation peaks)
   - `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
   - `--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3`
   - `--attention-config.flash_attn_version=2`, `--host 0.0.0.0`
7. **Streams `vllm.log`** until the API answers (≤10 min), then **warms the large-prefill path** with a ~26K-token request so the first real prompt doesn't stall on cold-serve compilation.

## Configuration

All knobs are environment variables, with sensible defaults:

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `8888` | API port |
| `MODEL_DIR` | `~/models/RadixArk/Qwen3.8-27B-NVFP4` | where weights live / get downloaded |
| `GPU_UTIL` | auto from live `nvidia-smi` | `--gpu-memory-utilization` override |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | leave it — MTP-3 OOMs otherwise |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | set `1` only if you have a CUDA toolkit with `nvcc` |

## Why the #40914 patch

Stock vLLM 0.27.1 captures the MTP verify step as a context-free first-chunk `flash_attn` **FULL cudagraph** (the capture dummy batch has `seq_len == query_len`). The replayed graph therefore **never reads the KV cache**, and MTP over 4-bit TurboQuant KV degenerates into repetition loops / broken tool calls (13/15 battery failures; upstream issue [#40880](https://github.com/vllm-project/vllm/issues/40880)).

The patch routes uniform K+1 spec-verify batches through the decode kernel with all-GPU synthetic args (same `synth_seq_lens` trick as the continuation path — no CPU-side per-request branching, so capture/replay stay valid). After the fix: **0/15 failures, ~160 tok/s**.

The patch is applied once per venv (a marker grep skips it on re-runs; a `.pre40914.bak` copy is kept). If the installed vLLM layout ever changes, the script refuses to patch blindly, tells you to check whether PR #40914 has merged upstream, and exits — because serving stock MTP×TurboQuant without the fix garbles output.

## Why concurrency is pinned to 1

`--max-num-seqs 1` is not a throttle — it's a hard necessity on this KV path, for two reasons:

1. **MTP × TurboQuant KV doesn't survive batching.** Per the comment in `start.sh`: *"MTP + concurrency crashes this KV path."* MTP-3 speculative decoding over the 4-bit TurboQuant KV cache is only stable with a single session; more sequences push the scheduler into mixed batches that this path doesn't handle (the same fragile territory as the [garbled-output bug](#why-the-40914-patch) — don't run concurrent load on it).
2. **The KV pool holds only ~1.09 full-context sessions.** `--kv-cache-memory-bytes 5905580032` fixes the pool at 5.5 GiB = 286,466 tokens (log: `GPU KV cache size: 286,466 tokens`, `Maximum concurrency for 262,144 tokens per request: 1.09x`). One 262,144-token request (plus MTP draft slots) consumes effectively the entire pool, so there is no room for a second session.

### Can I get more concurrency?

**Yes — shrink the per-request context ceiling.** The pool size stays fixed (5.5 GiB); what changes is how many requests fit in it. Rough ceiling from the KV budget:

| `--max-model-len` | KV pool per request | Concurrent ceiling |
|---|---|---|
| 262,144 (256K) | ~100% | ~1 |
| 131,072 (128K) | ~50% | ~2 |
| 65,536 (64K) | ~25% | ~4 |
| 32,768 (32K) | ~12.5% | ~8 |
| 16,384 (16K) | ~6% | ~17 |

Concretely, edit `start.sh` — for a 32K-context, ~8-session setup:

```bash
MAX_MODEL_LEN=32768
# in SERVE_ARGS: --max-num-seqs 8
```

Two caveats:

- **Stability first.** Raising `--max-num-seqs` while MTP-3 stays on is exactly the "MTP + concurrency" territory `start.sh` deliberately avoids. For real concurrent serving, plan on also dialing back or dropping speculation (`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`, or remove the flag) and re-running the garble battery.
- **The pin doesn't grow.** Lowering context adds no KV memory — it only shrinks each session's worst-case footprint. To increase the pool itself, raise `--kv-cache-memory-bytes` (which takes memory from weights/activations on the 32 GB card), and expect `--max-num-batched-tokens 512` to also need raising once several sessions decode together.

## Stopping

```bash
./stop.sh
```

Graceful (SIGTERM → SIGKILL after 15s), sweeps orphaned `VLLM::EngineCore` processes, verifies the port is free, and is idempotent — nothing to stop = it just says so. Manual equivalent:

```bash
kill $(cat vllm.pid)
pkill -f 'VLLM::EngineCore'
```

## Notes & troubleshooting

- **`python3-dev` not installed** — Triton's JIT C shim `#includes <Python.h>`. The script auto-installs a uv-managed CPython 3.14 and wraps `CC` with `~/.local/bin/cc-triton` so Triton finds `Python.h` without root.
- **No CUDA toolkit** — expected and fine; the flashinfer sampler JIT is disabled and MoE/linear kernels use the CUTLASS path.
- **First request after a cold boot** — the script pre-warms the large-prefill path; if you skipped warmup (`WARN warmup failed`), the first big prompt compiles on the spot and can take a minute.
- **Single session by design** — see [Why concurrency is pinned to 1](#why-concurrency-is-pinned-to-1); it can be raised at the cost of context length.
- **Desktop cohabitation** — GPU util is derived from *live free* memory, so the desktop's 1–2 GiB are respected. If you run headless, you can push `GPU_UTIL=0.9` or higher… you won't gain anything for a single stream.
- **Model download interrupted** — just re-run `./start.sh`; `snapshot_download` resumes.
- **Patch fails to apply** (`anchor not found`) — the installed vLLM no longer matches the 0.27.1 layout. Check whether PR #40914 has merged upstream before serving, since stock MTP×TurboQuant garbles output.

## Repository layout

```
.
├── .gitignore    # keeps .venv/, vllm.log, vllm.pid out of git
├── README.md
├── start.sh      # the whole setup: venv → patch → download → serve → warm
├── stop.sh       # graceful shutdown: TERM → KILL → orphan sweep → verify
├── vllm.log      # server log (runtime artifact)
├── vllm.pid      # server pid (runtime artifact)
└── .venv/        # created by start.sh (runtime artifact)
```

## References

- Model: [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
- vLLM: [vllm-project/vllm](https://github.com/vllm-project/vllm) — PR [#40914](https://github.com/vllm-project/vllm/pull/40914), issue [#40880](https://github.com/vllm-project/vllm/issues/40880)
