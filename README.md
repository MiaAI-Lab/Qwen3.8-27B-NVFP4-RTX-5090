# Qwen3.8-27B-NVFP4 on a Single RTX 5090 (vLLM)

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://ko-fi.com/Z8Z3SPLOD" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://storage.ko-fi.com/cdn/kofi6.png?v=6" alt="Buy Me a Coffee at ko-fi.com" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

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
| `PROFILE` | `default` | `default` = the profile above; `fast-128k` = see [Opt-in profile: fast-128k](#opt-in-profile-fast-128k) |

## Opt-in profile: fast-128k

```bash
PROFILE=fast-128k ./start.sh
```

Same stack, same patch, same single session; four values change. If you don't need more than 128K of context,
it decodes faster and prefills about twice as fast:

| | `default` | `fast-128k` |
|---|---|---|
| `--max-model-len` | 262144 | 131072 |
| `--kv-cache-memory-bytes` | 5905580032 (5.5 GiB, 286,466-token pool) | 3758096384 (3.5 GiB, 137,313-token pool) |
| MTP `num_speculative_tokens` | 3 | 6 |
| `--max-num-batched-tokens` | 512 | 2048 |
| `--compilation-config` | (vLLM default) | `{"compile_ranges_endpoints":[8,2048]}` |

Why each lever:

- **MTP-6.** On its own it gives +26–29% greedy decode. Mean accepted length per step goes from 2.9–3.6 to 3.0–5.7, depending on the prompt.
- **2048-token prefill chunks.** On their own they make prefill about 2x faster at 8K–64K.
- **The compile-range split.** With 2048-token chunks, the 7-token MTP-6 verify step shares one torch.compile range (1–2048) with prefill, and decode loses about 7–10%. The `[8, 2048]` endpoint gives steps of 8 tokens or fewer their own range and recovers it.
- **The 3.5 GiB pin.** It holds 128K plus the MTP-6 drafts. 3.0 GiB is too small for MTP-6 at 128K, and the server refuses to start. The smaller pin also leaves about 2 GB more for a desktop on the same card.

### Measured (RTX 5090, same box, same harness, same day)

Setup:
- `start.sh` @ `a5f9bff`, with only the four values above changed.
- Software: vLLM 0.27.1 + this repo's #40914 backport, torch 2.13.0+cu130, flashinfer-python 0.6.16.post3, nvidia-cutlass-dsl 4.6.0.
- Model: `RadixArk/Qwen3.8-27B-NVFP4` @ `319f741c`.
- Machine: Windows 11 + WSL2 Ubuntu (4 vCPU, 8 GB to the VM), WDDM driver 621.01. The 5090 also drives the desktop, which holds about 2 GB of VRAM.
- Each config is a fresh server start plus this repo's 26K warm-up, then the bench.

Bench: [`bench/bench.py`](bench/bench.py).
- Decode: 3 prompts (prose / code / JSON), 512 new tokens each, greedy, thinking off, streamed. TTFT is excluded. The value is the median over all requests.
- Prefill: 3 nonce-prefixed prompts per size, `max_tokens 1`, wall time median.

```bash
.venv/bin/python bench/bench.py --runs 5 --tokenizer "$MODEL_DIR/tokenizer.json" decode prefill
```

| Config | Decode median tok/s (n) | prose / code / json | Prefill 8K / 32K / 63.5K (s) | Whole-GPU MiB at load (incl. desktop, of 32,607) |
|---|---|---|---|---|
| `default` | 117.85 (9) | 82.5 / 119.6 / 120.7 | 1.56 / 6.81 / 16.99 | 30,968 MiB |
| `default`, re-run at the end (drift check) | 118.09 (9) | 84.6 / 121.4 / 119.0 | 1.66 / 7.06 / 16.76 | 30,553 MiB |
| MTP-6 only (256K, 5.5 GiB) | 152.20 (15) | 82.0 / 157.3 / 163.4 | 1.64 / 7.21 / 17.24 | 30,565 MiB |
| 2048 chunks only (256K, 5.5 GiB) | 114.19 (9) | 79.5 / 114.2 / 117.2 | 0.77 / 3.94 / 9.79 | 30,512 MiB |
| MTP-6 + 2048 + 3.5 GiB, no compile split | 138.17 (15) | 80.7 / 138.2 / 161.5 | 0.78 / 3.99 / 9.91 | 28,047 MiB |
| **`fast-128k`** | **151.07 (15)** (+28%) | 82.0 / 151.1 / 166.5 | **0.78 / 3.99 / 9.90** (1.7–2.0x) | **28,045 MiB** |

These are absolute numbers from one WSL2 box and this harness. They are lower than the ~160 tok/s quoted above, so compare the ratios.

`fast-128k` passed the same checks as `default`:
- Garble battery: 0/15 greedy and 0/15 sampled (T 0.7). The battery targets repetition loops and broken tool calls.
- Quality: 16/16 greedy (10 code tasks with unit tests, 6 exact answers).
- Tool calls: `tool_choice` auto and required both work, and so does a tool-result turn.
- Engine errors: 0 over the runs.

Not validated, so check before relying on it:
- **Long agentic sessions.** We ran the battery, 64K prefills and about an hour of benching, plus about 2 hours of live single-user chat traffic. We did not run long (>40K-token) multi-turn agent sessions.
- **Prose decode is flat.** It stays at about 82 tok/s; the gain is on code and structured output.
- **MTP-6 + 2048 chunks at the full 262K.** Not measured. MTP-6 alone does start at 262K with the 5.5 GiB pin (its pool is exactly 262,144 tokens). MTP-7 does not start at that pin.
- **Native Linux and headless cards.** Not measured; all the numbers above are from WSL2 with a desktop on the card.

The profile was measured and contributed by Yves La Rose (Nether Labs) on top of this recipe.

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
├── start.sh      # the whole setup: venv → patch → download → serve → warm (PROFILE=default|fast-128k)
├── bench/bench.py # single-stream decode + prefill bench behind the fast-128k table
├── stop.sh       # graceful shutdown: TERM → KILL → orphan sweep → verify
├── vllm.log      # server log (runtime artifact)
├── vllm.pid      # server pid (runtime artifact)
└── .venv/        # created by start.sh (runtime artifact)
```

## References

- Model: [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
- vLLM: [vllm-project/vllm](https://github.com/vllm-project/vllm) — PR [#40914](https://github.com/vllm-project/vllm/pull/40914), issue [#40880](https://github.com/vllm-project/vllm/issues/40880)
