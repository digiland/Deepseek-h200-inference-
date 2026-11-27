# Generate DeepSeek V4.1 Flash thinking-mode replies for drafter fine-tuning.
# usage: gen_replies.py <prompts.jsonl> <out.jsonl> <limit> <port> [<port> ...]
# Resumable: skips ids already in out.jsonl. Keeps exact prompt and reply token ids.
import asyncio, json, sys, time, aiohttp
src, out, limit, ports = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
CONC = 256  # per server copy
done = set()
try:
    for l in open(out): done.add(json.loads(l)["id"])
except FileNotFoundError:
    pass
todo = [json.loads(l) for l in open(src)][:limit]
todo = [p for p in todo if p["id"] not in done]
print(f"already {len(done)}, to do {len(todo)}", flush=True)
q = asyncio.Queue()
for p in todo: q.put_nowait(p)
stats = {"ok": 0, "err": 0, "tok": 0, "t0": time.time()}

async def worker(sess, port, fh):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    while not q.empty():
        p = q.get_nowait()
        body = {"model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
                "messages": [{"role": "user", "content": p["prompt"]}],
                "temperature": 0.6, "top_p": 0.95, "max_tokens": int(__import__("os").environ.get("MAX_TOKENS", "8192")),
                "return_token_ids": True}
        try:
            async with sess.post(url, json=body, timeout=aiohttp.ClientTimeout(total=3600)) as r:
                j = await r.json()
            c = j["choices"][0]; m = c["message"]
            rec = {"id": p["id"], "prompt": p["prompt"],
                   "reasoning": m.get("reasoning") or m.get("reasoning_content"),
                   "content": m.get("content"), "finish_reason": c.get("finish_reason"),
                   "prompt_token_ids": j.get("prompt_token_ids"), "token_ids": c.get("token_ids")}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            stats["ok"] += 1; stats["tok"] += j["usage"]["completion_tokens"]
        except Exception as e:
            stats["err"] += 1
            print("ERR", p["id"], repr(e)[:200], flush=True)
        if (stats["ok"] + stats["err"]) % 500 == 0:
            dt = time.time() - stats["t0"]
            print(f"{time.strftime('%H:%M:%S')} ok {stats['ok']} err {stats['err']} "
                  f"reply tokens {stats['tok']} ({stats['tok']/dt:.0f} tok/s)", flush=True)

async def main():
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as sess:
        with open(out, "a") as fh:
            await asyncio.gather(*[worker(sess, port, fh) for port in ports for _ in range(CONC)])
    dt = time.time() - stats["t0"]
    print(f"done ok {stats['ok']} err {stats['err']} reply tokens {stats['tok']} in {dt:.0f} s", flush=True)

asyncio.run(main())
