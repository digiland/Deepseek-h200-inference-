#!/usr/bin/env python3
"""Mean acceptance length at temperature 0 on short ShareGPT prompts (stdlib only).

Sends the first human turn of 80 fixed ShareGPT conversations (seed 7, prompts of 8-400 words),
greedy, max 256 tokens, thinking off, 8 at a time, and reads the server's spec-decode counters
before and after. Acceptance length = 1 + accepted draft tokens / draft steps (tokens emitted per
verify step, bonus token included).

usage: accept_probe.py <port> <tag> [n_prompts] [thinking 0|1] [short|long]
"""
import concurrent.futures as cf
import json
import random
import re
import sys
import urllib.request

port, tag = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 80
thinking = len(sys.argv) > 4 and sys.argv[4] == "1"
thinking_default = len(sys.argv) > 4 and sys.argv[4] == "d"  # as vllm bench: no kwargs
import os
TEMP = float(os.environ.get("PROBE_TEMP", "0"))
TOP_P = float(os.environ.get("PROBE_TOP_P", "1.0"))
SHAREGPT = os.environ.get("SHAREGPT", "/data/nvme3/lab/teachers/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json")


def metrics():
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as r:
        text = r.read().decode()
    vals = {}
    for key in ("spec_decode_num_drafts", "spec_decode_num_accepted_tokens",
                "spec_decode_num_draft_tokens"):
        tot = 0.0
        for m in re.finditer(rf"^vllm:{key}(?:_total)?(?:{{[^}}]*}})?\s+([\d.e+]+)$", text, re.M):
            tot += float(m.group(1))
        vals[key] = tot
    pos = {}
    for m in re.finditer(r'^vllm:spec_decode_num_accepted_tokens_per_pos(?:_total)?\{[^}]*position="(\d+)"[^}]*\}\s+([\d.e+]+)$', text, re.M):
        pos[int(m.group(1))] = pos.get(int(m.group(1)), 0.0) + float(m.group(2))
    vals["per_pos"] = pos
    return vals


long_mode = len(sys.argv) > 5 and sys.argv[5] == "long"
data = json.load(open(SHAREGPT))
rng = random.Random(7)
rng.shuffle(data)
prompts = []
for conv in data:
    turns = conv.get("conversations") or []
    if not turns or turns[0].get("from") != "human":
        continue
    if not long_mode:
        t = turns[0]["value"]
        if 8 <= len(t.split()) <= 400:
            prompts.append([{"role": "user", "content": t}])
    else:
        # Multi-turn history of 1500-5000 words ending on a human turn.
        msgs, words = [], 0
        for t in turns:
            role = "user" if t.get("from") == "human" else "assistant"
            msgs.append({"role": role, "content": t["value"]})
            words += len(t["value"].split())
        while msgs and msgs[-1]["role"] != "user":
            words -= len(msgs[-1]["content"].split())
            msgs.pop()
        if 1500 <= words <= 5000:
            prompts.append(msgs)
    if len(prompts) >= n:
        break


def ask(q):
    body = {"model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
            "messages": q, "temperature": TEMP, "top_p": TOP_P, "max_tokens": 256,
            "chat_template_kwargs": {"thinking": thinking, "enable_thinking": thinking}}
    if thinking_default:
        body.pop("chat_template_kwargs")
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return out["usage"]["completion_tokens"]


before = metrics()
with cf.ThreadPoolExecutor(8) as ex:
    toks = list(ex.map(ask, prompts))
after = metrics()
d = after["spec_decode_num_drafts"] - before["spec_decode_num_drafts"]
a = after["spec_decode_num_accepted_tokens"] - before["spec_decode_num_accepted_tokens"]
per = {k: after["per_pos"].get(k, 0) - before["per_pos"].get(k, 0) for k in after["per_pos"]}
rates = " ".join(f"p{k}={per[k] / d:.3f}" for k in sorted(per)) if d else ""
print(f"{tag} T={TEMP} top_p={TOP_P}: prompts={len(prompts)} out_tokens={sum(toks)} draft_steps={int(d)} "
      f"accepted={int(a)} accept_len={1 + a / d if d else float('nan'):.3f} "
      f"(accepted per step {a / d if d else float('nan'):.3f}) per-position: {rates}")
