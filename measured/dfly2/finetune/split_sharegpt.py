# Split ShareGPT by conversation id: 15% held out for benchmarks, 85% for drafter fine-tune prompts.
import json, hashlib, random
SRC = "/data/nvme3/lab/teachers/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
OUT = "/data/nvme3/dfly2/ft/data"
rows = json.load(open(SRC))
held, train = [], []
for r in rows:
    (held if int(hashlib.md5(r["id"].encode()).hexdigest(), 16) % 100 < 15 else train).append(r)
json.dump(held, open(f"{OUT}/sharegpt_heldout.json", "w"))
seen, prompts = set(), []
for r in train:
    turns = r.get("conversations") or []
    if not turns or turns[0].get("from") != "human":
        continue
    p = turns[0]["value"].strip()
    if not (20 <= len(p) <= 6000) or p in seen:
        continue
    seen.add(p); prompts.append({"id": r["id"], "prompt": p})
random.Random(7).shuffle(prompts)
with open(f"{OUT}/train_prompts.jsonl", "w") as f:
    for p in prompts: f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"conversations {len(rows)}, held out {len(held)}, train conversations {len(train)}, unique train prompts {len(prompts)}")
