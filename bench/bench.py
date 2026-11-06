#!/usr/bin/env python3
"""Single-stream decode + prefill bench for an OpenAI-compatible server (used for the README tables).

usage:
  .venv/bin/python bench/bench.py --base http://127.0.0.1:8888/v1 --model qwen38-nvfp4 \
      --tokenizer ~/models/RadixArk/Qwen3.8-27B-NVFP4/tokenizer.json --runs 5 decode prefill

decode : 3 fixed prompts (prose / code / json), greedy (temperature 0, top_k 1, seed 1234), thinking off,
         512 new tokens, streamed. One untimed warm-up, then --runs rounds. decode tok/s per request =
         (completion_tokens - 1) / (last content chunk - first content chunk), i.e. TTFT excluded.
prefill: deterministic pseudo-English prompts of ~8K / 32K / 63.5K tokens (3 each, every one starts with a
         unique nonce so no prefix cache can hit), max_tokens 1, non-streamed; reports wall time per request.
Prints per-request lines and a median summary. Stdlib only, except `tokenizers` for prefill prompt sizing
(installed in the vLLM / SGLang venv). Needs no API key unless --api-key is given.
"""
import argparse, http.client, json, random, statistics, sys, time, urllib.parse

DECODE = [
    ("D1-prose", "Write a long, detailed essay on the history of the printing press, from Gutenberg to modern "
                 "digital printing. Use several paragraphs."),
    ("D2-code", "Write a complete Python implementation of a red-black tree with insert, delete and search "
                "methods. Include docstrings and type hints."),
    ("D3-json", "Generate a JSON array of 40 fictional hardware-store products. Each object has id (integer), "
                "name, category, price (number) and tags (array of strings). Output only the JSON."),
]
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 1234}
PREFILL_TARGETS = (8192, 32768, 63488)

WORDS = """time year people way day man thing woman life child world school state family student group country
problem hand part place case week company system program question work government number night point home
water room mother area money story fact month lot right study book eye job word business issue side kind head
house service friend father power hour game line end member law car city community name president team minute
idea kid body information back parent face others level office door health person art war history party result
change morning reason research girl guy moment air teacher force education foot boy age policy process music
market sense nation plan college interest death experience effect class control care field development role
effort rate heart drug show leader light voice wife police mind price report decision son view relationship town
road arm difference value building action model season society tax director position player record paper space
ground form event official matter center couple site project activity star table need court oil situation cost
industry figure street image phone data picture practice piece land product doctor wall patient worker news test
movie north love support technology step baby computer type attention film tree source organization hair window
evidence population truth song energy unit bridge river engine signal harbor valve cable ledger winter summer
quickly slowly carefully often rarely always never usually quietly openly gently firmly briefly early late
build carry measure follow explain consider remember suggest describe develop include continue provide create
become report allow offer appear serve expect improve reduce prepare manage protect observe record compare
bright quiet heavy narrow ancient modern simple complex careful sudden steady distant local final central
and the of to in for with on at by from about over under between through during before after across""".split()


def _sentence(rng):
    n = rng.randint(8, 18)
    w = [rng.choice(WORDS) for _ in range(n)]
    w[0] = w[0].capitalize()
    if n > 10 and rng.random() < 0.4:
        w[rng.randint(3, n - 3)] += ","
    return " ".join(w) + "."


def _haystack(seed, approx_words):
    rng = random.Random(seed)
    paras, count = [], 0
    while count < approx_words:
        p = " ".join(_sentence(rng) for _ in range(rng.randint(4, 9)))
        paras.append(p)
        count += len(p.split())
    return "\n\n".join(paras)


def prefill_prompts(tokenizer_path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    ntoks = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    out = []
    for target in PREFILL_TARGETS:
        for r in range(1, 4):
            head = f"[bench nonce p{target}-r{r}-{random.Random(target * 10 + r).getrandbits(48):012x}]\n"
            tail = "\n\nReply with the single word OK."
            budget = target - 20 - ntoks(head) - ntoks(tail)
            text = _haystack(1000 + target + r, int(target * 0.95))
            enc = tok.encode(text, add_special_tokens=False)
            end = enc.offsets[budget - 1][1]
            dot = text.rfind(".", 0, end)          # cut at a sentence end
            out.append((target, r, head + (text[: dot + 1] if dot > 0 else text[:end]) + tail))
    return sorted(out, key=lambda e: (e[1], e[0]))


def chat(base, model, key, prompt, max_tokens, stream):
    u = urllib.parse.urlparse(base)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, **GREEDY}
    if stream:
        body.update(stream=True, stream_options={"include_usage": True})
    hdr = {"Content-Type": "application/json"}
    if key:
        hdr["Authorization"] = "Bearer " + key
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=1800)
    t0 = time.perf_counter()
    c.request("POST", u.path.rstrip("/") + "/chat/completions", body=json.dumps(body).encode(), headers=hdr)
    r = c.getresponse()
    if r.status != 200:
        raise SystemExit(f"HTTP {r.status}: {r.read()[:500]!r}")
    if not stream:
        d = json.loads(r.read())
        return {"total_s": time.perf_counter() - t0, "usage": d.get("usage") or {}}
    usage, t_first, t_last = {}, None, None
    for line in r:
        line = line.strip()
        if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
            continue
        d = json.loads(line[5:])
        usage = d.get("usage") or usage
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                t_last = time.perf_counter()
                t_first = t_first or t_last
    ct = int(usage.get("completion_tokens") or 0)
    rec = {"total_s": time.perf_counter() - t0, "ttft_s": (t_first - t0) if t_first else None, "usage": usage}
    if t_first and t_last and t_last > t_first and ct > 1:
        rec["decode_tps"] = (ct - 1) / (t_last - t_first)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="qwen38-nvfp4")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--tokenizer", help="tokenizer.json of the served model (prefill only)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("suites", nargs="+", choices=["decode", "prefill"])
    a = ap.parse_args()
    summary = {}
    if "decode" in a.suites:
        chat(a.base, a.model, a.api_key, DECODE[0][1], 64, True)          # untimed warm-up
        per = {}
        for run in range(1, a.runs + 1):
            for pid, prompt in DECODE:
                rec = chat(a.base, a.model, a.api_key, prompt, 512, True)
                per.setdefault(pid, []).append(rec.get("decode_tps") or 0.0)
                print(f"decode {pid} run {run}: {rec.get('decode_tps', 0):.2f} tok/s "
                      f"({rec['usage'].get('completion_tokens')} tokens)", flush=True)
        allv = [v for vs in per.values() for v in vs]
        summary["decode_median_tok_s"] = round(statistics.median(allv), 2)
        summary["decode_n"] = len(allv)
        summary.update({f"decode_{k}": round(statistics.median(v), 1) for k, v in per.items()})
    if "prefill" in a.suites:
        if not a.tokenizer:
            raise SystemExit("prefill needs --tokenizer path/to/tokenizer.json")
        per = {}
        for target, r, text in prefill_prompts(a.tokenizer):
            rec = chat(a.base, a.model, a.api_key, text, 1, False)
            per.setdefault(target, []).append(rec["total_s"])
            print(f"prefill {target} run {r}: {rec['total_s']:.2f} s "
                  f"({rec['usage'].get('prompt_tokens')} prompt tokens)", flush=True)
        summary.update({f"prefill_{k}_s": round(statistics.median(v), 2) for k, v in per.items()})
    print("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
