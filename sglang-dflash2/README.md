# Variant: SGLang 0.5.23 + DFlash2 on a single RTX 5090

This variant serves the same weights as the root recipe, **RadixArk/Qwen3.8-27B-NVFP4**, on a different engine: [SGLang](https://github.com/sgl-project/sglang) 0.5.23 with the **DFlash2** block-diffusion drafter ([z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)). It uses fp8 KV and SGLang's radix (prefix) cache, and it runs a single session on one 32 GB card that can still drive a desktop.

Compared with the root vLLM recipe on the same box:
- Decode is about **2x faster**: 226–249 tok/s greedy against 118.
- On repeated prompt prefixes, the second turn of a chat starts in about **0.1 s**. vLLM's prefix caching is off in the root recipe because of TurboQuant (vllm#46460), so there it starts in about 1.6 s.
- The cost is context. This variant runs **48K** of context, against 256K for the root recipe.

```bash
cd sglang-dflash2
./start.sh          # first run: venv + lock install, downloads, serve on :8888, warm-up
./stop.sh
```

Same API as the root recipe: `http://127.0.0.1:8888/v1`, model `qwen38-nvfp4`, thinking on by default. The model directory is shared with `../start.sh` (`MODEL_DIR`), so the 22 GiB download happens once. Only one of the two servers can run at a time on a 32 GB card.

## Pins

| | |
|---|---|
| Engine | `sglang==0.5.23`, `sglang-kernel==0.4.10`, `torch==2.14.1` (CUDA 13.0), `flashinfer-python==0.7.2` + `flashinfer-jit-cache==0.7.2` (cu130 index), `triton==3.8.0`, `transformers==5.19.0`. The full lock (`requirements.lock.txt`, 207 packages) is installed into a uv-managed Python 3.12 venv. |
| Target | `RadixArk/Qwen3.8-27B-NVFP4` @ `319f741cce68d7914884900c138a1fbb70a42f30` (the FP4 `lm_head` export) |
| Drafter | `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` (a mirror of `incoai/Qwen3.8-27B-DFlash2`) |
| CUDA toolkit | Our runs had CUDA 13.0 (`nvcc`) on PATH. `start.sh` uses `/usr/local/cuda-13.0` if it exists. We have not tested without it, because the prebuilt jit-cache should make it unnecessary. |

## Flags, and why

```
--tp-size 1 --max-running-requests 1
--reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-metrics
--kv-cache-dtype fp8_e4m3 --attention-backend flashinfer
--context-length 49152 --max-total-tokens 53248 --chunked-prefill-size 1024
--mamba-radix-cache-strategy extra_buffer_lazy --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 8
--mem-fraction-static 0.92
--speculative-algorithm DFLASH --speculative-draft-model-path <drafter>
--speculative-dflash-block-size 8 --speculative-draft-window-size 8192
--speculative-draft-model-quantization fp8
--language-only --mm-feature-transport cpu
env: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_COMPILE_THREADS=1
```

Most of these come from the [SGLang Qwen3.8-27B cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)'s RTX 5090 pins: FlashInfer attention on sm_120, 1024-token chunks next to a drafter, `extra_buffer_lazy`, and bf16 GDN state. We changed four things for a desktop-shared card:

- **`--speculative-draft-model-quantization fp8`.** It halves the drafter, from about 4.5 GB to 2.3 GB.
  - Decode cost about 6% in our runs: 262 tok/s with a bf16 drafter, 246 with fp8.
  - With the bf16 drafter, the KV pool left on a desktop-shared card was 12K tokens, smaller than one 14K-token prompt.
- **`--max-total-tokens 53248`.** `--mem-fraction-static` is a fraction of the memory that is *free at start*, so without a pin the desktop decides the pool size and the headroom. The pin makes every start take the same VRAM.
- **`--context-length 49152`.** This is what that pool can serve with the drafter's window.
- **`--language-only --mm-feature-transport cpu`.** These keep the vision tower off the GPU. This variant is text-only.

Headless card, or one with less desktop use: `MEM_FRACTION=0.95 CONTEXT_LENGTH=65536 MAX_TOTAL_TOKENS= ./start.sh` gave a 72,310-token pool on our box. The "finalist" row below was measured with those values, but it left only 0.94 GB of VRAM free.

## Measured (RTX 5090, same box, same harness, same day)

Setup:
- Machine: Windows 11 + WSL2 Ubuntu (4 vCPU, 8 GB to the VM), WDDM driver 621.x. The card also drives the desktop (about 2–2.6 GB of VRAM).
- Each config is a fresh start plus the 26K warm-up, then the benches.

Benches:
- Decode / prefill: [`../bench/bench.py`](../bench/bench.py). Decode is 3 prompts (prose / code / JSON) × 5 runs, 512 tokens each, greedy, thinking off, streamed, TTFT excluded, median over all requests. Prefill is the wall time of `max_tokens 1` requests on nonce-prefixed prompts.
- Two-turn: [`two_turn.py`](two_turn.py). A 14.4K-token fixed system prompt, then 2 turns, 3 repetitions, medians.

```bash
../.venv/bin/python ../bench/bench.py --runs 5 --tokenizer "$MODEL_DIR/tokenizer.json" decode prefill   # or sglang-dflash2/.venv/bin/python
python3 two_turn.py
```

| Config | Decode median tok/s (n) | prose / code / json | Prefill 8K / 32K / 63.5K (s) | Two-turn t1 / t2 total (s) [TTFT] | Whole-GPU MiB loaded-idle / peak (incl. desktop, of 32,607) |
|---|---|---|---|---|---|
| root `start.sh` (vLLM, MTP-3, 256K) | 117.85 (9) | 82.5 / 119.6 / 120.7 | 1.56 / 6.81 / 16.99 | — | 31,014 / 31,039 |
| root `PROFILE=fast-128k` (vLLM, MTP-6) | 151.07 (15) | 82.0 / 151.1 / 166.5 | 0.78 / 3.99 / 9.90 | 4.92 / 4.59 [1.59 / 1.56] | 28,447 / 28,793 |
| SGLang DFlash2 finalist (mem 0.95, ctx 64K, unpinned 72K pool) | **249.19 (15)** | 129.5 / 259.2 / 249.2 | 0.78 / 4.35 / 11.58 | **2.52 / 1.08 [0.19 / 0.09]** | 31,643 / 31,767 |
| **this `start.sh`** (mem 0.92, ctx 48K, pool pinned 53,248) | **226.32 (15)** | 116.6 / 244.4 / 235.2 | 0.88 / 4.98 / >ctx | 3.36 / 1.93 [0.29 / 0.10] | 29,946 / 31,680 |

How to read the table:
- Two-turn totals depend on how long the sampled answer is: one t2 hit the 600-token cap. TTFT is the stable signal.
- During the pinned run (bottom row), the desktop was using 10–13% of the GPU and an extra 2 GB of VRAM. With a pinned pool that slows decode rather than failing it: 226 vs the finalist's 249.

Quality and correctness, using our batteries (not in this repo):
- Greedy quality is 16/16 (10 code tasks with unit tests, 6 exact answers).
- Garble is 0/15 greedy + 0/15 sampled (T 0.7). The garble battery checks for repetition loops and broken tool calls.
- These were measured on the finalist and on this config with an unpinned pool. On the pinned config we ran decode, prefill, a prefix-cache correctness check (a cached prefix answers the same as a cold one), and the tool-call checks.
- Tool calls work with `tool_choice` auto and required, and in a tool-result turn. `reasoning_effort` none / low / medium / xhigh all work.
- The draft is lossless: greedy output is the target's. Mean accepted length on chat traffic was 2.6–3.7 tokens per verify step, against 2.4 for vLLM MTP-6 on the same turns.

## Where the time goes (decode, finalist)

- About 46 verify steps/s, about 21.6 ms per step.
- Prose accepts about 2.8 tokens per step and code about 5.6. Step time is nearly constant, so the tok/s gap between prose and code is acceptance.
- Each step reads about 20.1–20.7 GB:
  - target body + FP4 `lm_head`: 17.61 GB
  - fp8 drafter: 2.31 GB
  - GDN state: 0.16–0.8 GB
  - KV
- That is about 0.93–0.96 TB/s, or **52–53% of the 5090's 1.79 TB/s**. The rest of each step is not memory traffic. Likely causes:
  - the Triton GDN kernels on sm_120 (the FlashInfer GDN fast path is sm_100-only)
  - drafter compute
  - scheduler CPU on 4 vCPUs
  - WDDM

  A native-Linux box would show how much of this is the platform.

## Caveats / not validated

- **Context.** It is 48K, not 256K. A single prompt above the pool is rejected.
- **Load.** Long agentic sessions were not stress-tested beyond the batteries and a few hours of single-user traffic. Neither was concurrency: everything here is `--max-running-requests 1`.
- **Desktop VRAM.** Whole-GPU use reached 31,680 of 32,607 MiB once the desktop grew. A game on the same card will contend, and WDDM pages memory rather than failing, so serving just gets slow.
- **WSL2 RAM.** MemAvailable was 1.6–2.0 GB of the 8 GB VM while serving. Don't run two engines at once.
- **Untested setups.** We did not test native Linux, headless cards, or a box without `nvcc`.

## Credits

- **Mia's AI Lab**: this repo's recipe, which made this comparison possible. The model choice, the warm-up, the desktop-cohabitation approach and the bench conventions all come from the root `start.sh`.
- **Z Lab / Inco AI**: DFlash 2 and the Qwen3.8-27B drafter (Zhijian Liu, [@zhijianliu](https://x.com/zhijianliu), and team): [z-lab/dflash](https://github.com/z-lab/dflash), [blog](https://inco.ai/blog/dflash2/).
- **SGLang**: the [Qwen3.8-27B cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B) RTX 5090 pins this variant starts from.
- **Prior art on the same card:** [darksidewalker/qwen3.8-27b-sglang-dspark-blackwell](https://github.com/darksidewalker/qwen3.8-27b-sglang-dspark-blackwell) and [shiftedx/Qwen3.8-27B-RTX5090-SGLang-DFlash](https://github.com/shiftedx/Qwen3.8-27B-RTX5090-SGLang-DFlash). Both are Docker-based; this variant is a plain venv next to the vLLM recipe.
- Measured and contributed by Yves La Rose (Nether Labs).
