#!/usr/bin/env python3
"""Quick quality probe for a running DeepSeek-V4.1-Flash server (stdlib only).

50 fixed multiplication/addition questions with known answers, greedy decoding,
thinking off. Prints accuracy and writes every answer to <out>.json so two server
variants can be compared answer by answer (kernel or quantisation changes that break
the numerics show up as lost accuracy or changed answers).

usage: quality_v41.py <port> <tag> <out_dir>
"""
import concurrent.futures as cf
import json
import random
import re
import sys
import urllib.request

port, tag, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
rng = random.Random(41)
qs = []
for i in range(50):
    a, b = rng.randint(12, 99), rng.randint(12, 99)
    if i % 2:
        c = rng.randint(100, 999)
        qs.append((f"What is {a} * {b} + {c}? Reply with the number only.", a * b + c))
    else:
        qs.append((f"What is {a} * {b}? Reply with the number only.", a * b))


def ask(q):
    body = {
        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "messages": [{"role": "user", "content": q}],
        "temperature": 0, "max_tokens": 32,
        "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
    }
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            m = json.loads(r.read())["choices"][0]["message"]
            return (m.get("content") or "") + "|" + (m.get("reasoning_content") or "")
    except Exception as e:  # noqa: BLE001
        return f"ERROR {e}"


with cf.ThreadPoolExecutor(16) as ex:
    answers = list(ex.map(ask, [q for q, _ in qs]))
ok = 0
rows = []
for (q, want), ans in zip(qs, answers):
    nums = re.findall(r"-?\d+", ans.split("|")[0].replace(",", ""))
    good = bool(nums) and int(nums[0]) == want
    ok += good
    rows.append({"q": q, "want": want, "answer": ans, "correct": good})
json.dump({"tag": tag, "accuracy": ok / len(qs), "rows": rows},
          open(f"{out_dir}/quality-{tag}.json", "w"), indent=1)
print(f"  quality {tag}: {ok}/{len(qs)} correct")
