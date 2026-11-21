#!/usr/bin/env python3
"""Output-quality gate for a running OpenAI-compatible server.

Two checks, both deterministic (temperature 0):

1. logprobs:  64 fixed prompts, 64 greedy tokens each with top-5 logprobs.
   Saved to --save; with --ref (an earlier --save from the reference profile)
   it reports greedy-token agreement, first-divergence position and mean
   |delta logprob| on the reference's chosen tokens. This catches numeric drift
   (e.g. the FP8 dense path) long before task accuracy moves.
2. gsm8k:     N grade-school math questions via /v1/chat/completions; accuracy
   from the last number in the answer. This catches real quality loss.

    python3 bench/quality_gate.py --save results/vllm-dep4-mega-spec/quality.json
    python3 bench/quality_gate.py --save results/vllm-max/quality.json \\
        --ref results/vllm-dep4-mega-spec/quality.json

Exit code 1 if --ref is given and a gate fails: GSM8K drops by more than --max-acc-drop
points, or token agreement falls below --min-agreement.
Needs only aiohttp (present in the vLLM image). GSM8K data is fetched from GitHub
once, or passed with --gsm8k-path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import urllib.request
from pathlib import Path

import aiohttp

GSM8K_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/master/"
             "grade_school_math/data/test.jsonl")

# Diverse, fixed prompts: code, math, prose, multilingual, structured output, long-ish context.
PROMPTS = [
    "def quicksort(arr):\n    \"\"\"Sort a list using quicksort.\"\"\"\n",
    "The three laws of thermodynamics can be summarized as follows:",
    "Translate into French: 'The library opens at nine and closes at five on weekdays.'\n",
    "Q: A train travels 180 km in 2.5 hours. What is its average speed in km/h?\nA:",
    "Write a JSON object describing a book with fields title, author, year, tags:\n",
    "SELECT customers.name, COUNT(orders.id) FROM customers",
    "In 1905, Albert Einstein published four papers that",
    "Explain the difference between TCP and UDP in two sentences.",
    "以下是关于长城的简要介绍：",
    "fn main() {\n    let v: Vec<i32> = (1..=10).collect();\n",
    "Recipe for a simple tomato soup:\n1.",
    "The derivative of x^3 * sin(x) with respect to x is",
    "Dear hiring manager,\n\nI am writing to apply for",
    "Summarize the plot of Romeo and Juliet in one paragraph:",
    "import numpy as np\n\ndef softmax(x):\n",
    "List five prime numbers greater than 100:",
]
# 64 prompts: the 16 above, plus each with a short distinct preamble so prefix caching
# cannot make them identical requests.
PROMPTS = [f"[{i}] {p}" if i >= 16 else p for i, p in enumerate(PROMPTS * 4)]


async def post(session, url, payload, sem, retries=3):
    async with sem:
        for attempt in range(retries):
            try:
                async with session.post(url, json=payload,
                                        timeout=aiohttp.ClientTimeout(total=1800)) as r:
                    r.raise_for_status()
                    return await r.json()
            except Exception as e:  # noqa: BLE001
                if attempt == retries - 1:
                    return {"error": str(e)}
                await asyncio.sleep(2 * (attempt + 1))


async def run_logprobs(base, model, conc):
    sem = asyncio.Semaphore(conc)
    async with aiohttp.ClientSession() as s:
        tasks = [post(s, f"{base}/v1/completions",
                      {"model": model, "prompt": p, "max_tokens": 64, "temperature": 0,
                       "logprobs": 5, "seed": 0}, sem) for p in PROMPTS]
        res = await asyncio.gather(*tasks)
    out = []
    for p, r in zip(PROMPTS, res):
        if "error" in r or not r.get("choices"):
            out.append({"prompt": p, "error": r.get("error", "no choices")})
            continue
        lp = r["choices"][0].get("logprobs") or {}
        out.append({"prompt": p, "tokens": lp.get("tokens") or [],
                    "token_logprobs": lp.get("token_logprobs") or [],
                    "top_logprobs": lp.get("top_logprobs") or []})
    return out


def compare_logprobs(cur, ref):
    agree, total, first_div, dlp = 0, 0, [], []
    for c, r in zip(cur, ref):
        if "error" in c or "error" in r:
            continue
        ct, rt = c["tokens"], r["tokens"]
        n = min(len(ct), len(rt))
        div = next((i for i in range(n) if ct[i] != rt[i]), n)
        first_div.append(div)
        agree += div
        total += len(rt)
        # |delta logprob| of the reference's chosen tokens, up to the first divergence
        for i in range(div):
            a, b = c["token_logprobs"][i], r["token_logprobs"][i]
            if a is not None and b is not None:
                dlp.append(abs(a - b))
    return {
        "prefix_agreement": agree / total if total else 0.0,
        "median_first_divergence": sorted(first_div)[len(first_div) // 2] if first_div else 0,
        "mean_abs_dlogprob": sum(dlp) / len(dlp) if dlp else float("nan"),
        "prompts_compared": len(first_div),
    }


def load_gsm8k(path, n):
    if path:
        text = Path(path).read_text()
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "gsm8k_test.jsonl"
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(GSM8K_URL, cache)
        text = cache.read_text()
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return rows[:n]


NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(s):
    m = NUM_RE.findall(s or "")
    if not m:
        return None
    try:
        return float(m[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


async def run_gsm8k(base, model, rows, conc, max_tokens):
    sem = asyncio.Semaphore(conc)
    async with aiohttp.ClientSession() as s:
        tasks = [post(s, f"{base}/v1/chat/completions", {
            "model": model, "temperature": 0, "max_tokens": max_tokens, "seed": 0,
            "messages": [{"role": "user", "content":
                          r["question"] + "\nSolve step by step. End with: The answer is <number>."}],
        }, sem) for r in rows]
        res = await asyncio.gather(*tasks)
    correct, errors = 0, 0
    for r, out in zip(rows, res):
        gold = float(r["answer"].split("####")[-1].strip().replace(",", ""))
        if "error" in out or not out.get("choices"):
            errors += 1
            continue
        pred = last_number(out["choices"][0]["message"].get("content"))
        correct += pred is not None and math.isclose(pred, gold, rel_tol=0, abs_tol=1e-6)
    return {"accuracy": correct / len(rows), "n": len(rows), "errors": errors}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=f"http://127.0.0.1:{os.environ.get('PORT', '8000')}")
    ap.add_argument("--model", default=os.environ.get("MODEL", "deepseek-ai/DeepSeek-V4.1-Flash"))
    ap.add_argument("--save", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--gsm8k-n", type=int, default=400)
    ap.add_argument("--gsm8k-path", default=None)
    ap.add_argument("--gsm8k-max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--max-acc-drop", type=float, default=2.0, help="GSM8K points")
    ap.add_argument("--min-agreement", type=float, default=0.3,
                    help="greedy-token prefix agreement vs --ref. Informative more than decisive: "
                         "TP vs DP layouts already diverge numerically; GSM8K is the real gate")
    args = ap.parse_args()

    result = {"logprobs": asyncio.run(run_logprobs(args.base_url, args.model, args.concurrency))}
    if args.gsm8k_n > 0:
        rows = load_gsm8k(args.gsm8k_path, args.gsm8k_n)
        result["gsm8k"] = asyncio.run(run_gsm8k(args.base_url, args.model, rows,
                                                args.concurrency, args.gsm8k_max_tokens))
        print(f"GSM8K accuracy {100 * result['gsm8k']['accuracy']:.1f}% "
              f"(n={result['gsm8k']['n']}, errors={result['gsm8k']['errors']})")
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    failed = False
    if args.ref:
        ref = json.loads(Path(args.ref).read_text())
        cmp = compare_logprobs(result["logprobs"], ref["logprobs"])
        result["vs_ref"] = {"ref": args.ref, **cmp}
        print(f"vs {args.ref}: greedy prefix agreement {100 * cmp['prefix_agreement']:.1f}%, "
              f"median first divergence at token {cmp['median_first_divergence']}, "
              f"mean |dlogprob| {cmp['mean_abs_dlogprob']:.4f}")
        if cmp["prefix_agreement"] < args.min_agreement:
            print(f"FAIL: agreement below {args.min_agreement:.0%}")
            failed = True
        if "gsm8k" in result and "gsm8k" in ref:
            drop = 100 * (ref["gsm8k"]["accuracy"] - result["gsm8k"]["accuracy"])
            result["vs_ref"]["gsm8k_drop_points"] = drop
            print(f"GSM8K change vs ref: {-drop:+.1f} points")
            if drop > args.max_acc_drop:
                print(f"FAIL: GSM8K dropped more than {args.max_acc_drop} points")
                failed = True
        result["vs_ref"]["passed"] = not failed
    Path(args.save).write_text(json.dumps(result, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
