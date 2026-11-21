#!/usr/bin/env python3
"""Prefix-affinity router for a vLLM data-parallel server (DP4 on one node).

With DP4 every rank has its own prefix cache, and vLLM's built-in balancer picks
the least-loaded rank per request. A multi-turn conversation then lands on a warm
rank only about 1/4 of the time on turn 2. This proxy pins each conversation to a
rank by setting `X-data-parallel-rank`, which both vLLM's Python and Rust
frontends honour. Load stays bounded: rendezvous hashing picks the conversation's
preferred rank unless that rank has more than (1 + SLACK) x the mean in-flight
requests, in which case the next rank in its hash order is used.

Affinity key, first match wins:
  1. header X-Session-ID (vLLM's multi-turn benchmark sends it with --send-conversation-id)
  2. the request's "user" field
  3. chat: hash of the first two messages (system + first user turn)
     completions: hash of the first PREFIX_CHARS characters / 512 token ids

    python3 serve/router/affinity_router.py --upstream http://127.0.0.1:8000 --port 8080 --dp 4

Clients then talk to :8080. Everything except the two generation endpoints is
proxied unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math

import aiohttp
from aiohttp import web

GEN_PATHS = {"/v1/completions", "/v1/chat/completions"}
HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "keep-alive"}


def affinity_key(headers, body: dict, prefix_chars: int) -> str | None:
    sid = headers.get("X-Session-ID")
    if sid:
        return f"s:{sid}"
    if body.get("user"):
        return f"u:{body['user']}"
    msgs = body.get("messages")
    if isinstance(msgs, list) and msgs:
        head = json.dumps(msgs[:2], sort_keys=True, ensure_ascii=False)[:prefix_chars]
        return "m:" + head
    prompt = body.get("prompt")
    if isinstance(prompt, str):
        return "p:" + prompt[:prefix_chars]
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
        return "t:" + ",".join(map(str, prompt[:512]))
    return None


def rendezvous_order(key: str, n: int) -> list[int]:
    def score(r):
        return hashlib.blake2b(f"{key}|{r}".encode(), digest_size=8).digest()
    return sorted(range(n), key=score, reverse=True)


class Router:
    def __init__(self, upstream: str, dp: int, slack: float, prefix_chars: int):
        self.upstream = upstream.rstrip("/")
        self.dp = dp
        self.slack = slack
        self.prefix_chars = prefix_chars
        self.inflight = [0] * dp
        self.routed = [0] * dp
        self.spilled = 0
        self.session: aiohttp.ClientSession | None = None

    async def start(self, app):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_read=3600),
            connector=aiohttp.TCPConnector(limit=0))

    async def stop(self, app):
        await self.session.close()

    def pick(self, key: str | None) -> int:
        cap = math.ceil((sum(self.inflight) / self.dp) * (1 + self.slack)) + 2
        if key is None:
            return min(range(self.dp), key=lambda r: self.inflight[r])
        order = rendezvous_order(key, self.dp)
        for i, r in enumerate(order):
            if self.inflight[r] < cap:
                self.spilled += i > 0
                return r
        return min(range(self.dp), key=lambda r: self.inflight[r])

    async def handle(self, request: web.Request) -> web.StreamResponse:
        raw = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
        rank = None
        if request.method == "POST" and request.path in GEN_PATHS:
            try:
                body = json.loads(raw)
            except ValueError:
                body = {}
            rank = self.pick(affinity_key(request.headers, body, self.prefix_chars))
            headers["X-data-parallel-rank"] = str(rank)
            self.inflight[rank] += 1
            self.routed[rank] += 1
        try:
            async with self.session.request(request.method, self.upstream + request.path_qs,
                                            data=raw, headers=headers) as up:
                resp = web.StreamResponse(status=up.status, headers={
                    k: v for k, v in up.headers.items() if k.lower() not in HOP_HEADERS})
                await resp.prepare(request)
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
        finally:
            if rank is not None:
                self.inflight[rank] -= 1

    async def stats(self, request):
        return web.json_response({"inflight": self.inflight, "routed": self.routed,
                                  "spilled": self.spilled})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--dp", type=int, default=4)
    ap.add_argument("--slack", type=float, default=0.25,
                    help="allowed in-flight excess over the mean before spilling to the next rank")
    ap.add_argument("--prefix-chars", type=int, default=2048)
    args = ap.parse_args()
    r = Router(args.upstream, args.dp, args.slack, args.prefix_chars)
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.on_startup.append(r.start)
    app.on_cleanup.append(r.stop)
    app.router.add_get("/router/stats", r.stats)
    app.router.add_route("*", "/{tail:.*}", r.handle)
    web.run_app(app, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
