#!/usr/bin/env python3
"""Two-turn chat with a large fixed system prompt: shows what the prefix (radix) cache buys.

usage: python3 two_turn.py [--base http://127.0.0.1:8888/v1] [--model qwen38-nvfp4] [--api-key KEY]

A ~14.4K-token system prompt (an agent's rules, identical every turn), then turn 1, then turn 2 replaying turn 1
and its answer. reasoning_effort low, temperature 0.6, top_p 0.95, max 600 new tokens, streamed; 3 repetitions.
Prints total time and TTFT per turn and the medians. Totals vary with the sampled answer length; TTFT does not.
"""
import argparse, json, statistics, time, urllib.request

PARA = ("You are the sysadmin agent for a Windows workstation. Follow the runbook: check services, disks, updates, "
        "scheduled tasks and GPU state before acting; never reboot without approval; report results briefly. ")
SYSTEM = "\n".join(f"[rule {i}] " + PARA for i in range(300))


def turn(a, messages):
    body = {"model": a.model, "messages": messages, "max_tokens": 600, "temperature": 0.6, "top_p": 0.95,
            "reasoning_effort": "low", "stream": True, "stream_options": {"include_usage": True}}
    hdr = {"Content-Type": "application/json", **({"Authorization": "Bearer " + a.api_key} if a.api_key else {})}
    req = urllib.request.Request(a.base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers=hdr)
    t0, first, text, usage = time.perf_counter(), None, [], {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for ch in d.get("choices") or []:
                dl = ch.get("delta") or {}
                if (dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning")) and first is None:
                    first = time.perf_counter() - t0
                if dl.get("content"):
                    text.append(dl["content"])
    return {"total_s": time.perf_counter() - t0, "ttft_s": first, "prompt_tokens": usage.get("prompt_tokens"),
            "content": "".join(text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="qwen38-nvfp4")
    ap.add_argument("--api-key", default="")
    a = ap.parse_args()
    res = {"t1": [], "t2": []}
    for rep in range(3):
        m = [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": f"(run {rep}) The D: drive is at 91% full. What do you check first, in 3 bullets?"}]
        r1 = turn(a, m)
        m += [{"role": "assistant", "content": r1["content"]},
              {"role": "user", "content": "Now give the single PowerShell command for the first check."}]
        r2 = turn(a, m)
        for k, r in (("t1", r1), ("t2", r2)):
            res[k].append(r)
            print(f"rep {rep} {k}: total {r['total_s']:.2f} s, TTFT {r['ttft_s'] or 0:.2f} s, "
                  f"{r['prompt_tokens']} prompt tokens", flush=True)
    for k, rs in res.items():
        print(f"{k} median: total {statistics.median(r['total_s'] for r in rs):.2f} s, "
              f"TTFT {statistics.median(r['ttft_s'] or 0 for r in rs):.2f} s")


if __name__ == "__main__":
    main()
