#!/usr/bin/env python3
"""Summarize a sweep directory, or compare two.

    python bench/summarize.py results/vllm-baseline
    python bench/summarize.py results/vllm-baseline results/vllm-max     # speedup table

Reads `<scenario>_c<conc>[_r<k>].json` (vllm bench serve) and
`<...>.multiturn.json` (vLLM multi-turn benchmark). Repeats are averaged
(the min-max spread is shown). Server counters come from the
`<name>.before.prom` / `<name>.after.prom` scrapes around each run, so the
prefix-cache hit rate and DSpark acceptance belong to that run only.
NUM_GPUS (default 4) sets the per-GPU column.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
from pathlib import Path

NUM_GPUS = int(os.environ.get("NUM_GPUS", "4"))
NAME_RE = re.compile(r"^(?P<sc>[a-z]+)_c(?P<c>\d+)(?:_r(?P<r>\d+))?(?P<mt>\.multiturn)?\.json$")


def parse_prom(path: Path) -> dict[str, float]:
    tot: dict[str, float] = {}
    if not path.exists():
        return tot
    for line in path.read_text().splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name_labels, _, val = line.rpartition(" ")
        name = name_labels.split("{", 1)[0]
        if name.endswith("_created"):
            continue
        try:
            tot[name] = tot.get(name, 0.0) + float(val)
        except ValueError:
            pass
    return tot


def _get(d, *names):
    for n in names:
        if n in d:
            return d[n]
    return None


def counters(before: Path, after: Path) -> dict[str, float | None]:
    """Per-run deltas of vLLM counters (SGLang gauges are read from `after`)."""
    a, b = parse_prom(after), parse_prom(before)

    def delta(*names):
        x, y = _get(a, *names), _get(b, *names)
        return None if x is None else x - (y or 0.0)

    q = delta("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries")
    h = delta("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits")
    drafts = delta("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_drafts")
    acc = delta("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_accepted_tokens")
    out = {
        "hit_rate": (h / q) if q else None,
        "accept_len": (1 + acc / drafts) if drafts else None,
    }
    if out["hit_rate"] is None and "sglang:cache_hit_rate" in a:
        out["hit_rate"] = a["sglang:cache_hit_rate"]
    if out["accept_len"] is None and "sglang:spec_accept_length" in a:
        out["accept_len"] = a["sglang:spec_accept_length"]
    return out


def load_one(f: Path, multiturn: bool) -> dict | None:
    try:
        d = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if multiturn:
        if not d:
            return None
        start = min(r["start_time_ms"] for r in d)
        end = max(r["start_time_ms"] + r["latency_ms"] for r in d)
        dur = max(1e-9, (end - start) / 1000)
        out_tok = sum(r["output_num_tokens"] for r in d)
        in_tok = sum(r["input_num_tokens"] for r in d)
        return {
            "output_throughput": out_tok / dur,
            "total_token_throughput": (out_tok + in_tok) / dur,
            "median_ttft_ms": statistics.median(r["ttft_ms"] for r in d),
            "p99_ttft_ms": sorted(r["ttft_ms"] for r in d)[int(0.99 * (len(d) - 1))],
            "median_tpot_ms": statistics.median(r["tpot_ms"] for r in d),
            "p99_tpot_ms": sorted(r["tpot_ms"] for r in d)[int(0.99 * (len(d) - 1))],
            "completed": len(d), "failed": 0,
        }
    keys = ["output_throughput", "total_token_throughput", "median_ttft_ms", "p99_ttft_ms",
            "median_tpot_ms", "p99_tpot_ms", "completed", "failed"]
    return {k: d.get(k) for k in keys}


def load(run_dir: Path) -> dict[tuple[str, int], dict]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for f in sorted(run_dir.iterdir()):
        m = NAME_RE.match(f.name)
        if not m:
            continue
        r = load_one(f, bool(m["mt"]))
        if r is None or not r.get("output_throughput"):
            continue
        stem = f.name.removesuffix(".json").removesuffix(".multiturn")
        r.update(counters(run_dir / f"{stem}.before.prom", run_dir / f"{stem}.after.prom"))
        groups.setdefault((m["sc"], int(m["c"])), []).append(r)
    rows = {}
    for k, rs in groups.items():
        agg = {}
        for field in rs[0]:
            vals = [r[field] for r in rs if isinstance(r.get(field), (int, float))]
            agg[field] = statistics.mean(vals) if vals else None
        tp = [r["output_throughput"] for r in rs]
        agg["n"] = len(rs)
        agg["spread"] = (max(tp) - min(tp)) / statistics.mean(tp) if len(rs) > 1 else None
        rows[k] = agg
    return rows


def fmt(v, nd=1, pct=False):
    if v is None:
        return "-"
    return f"{100 * v:.0f}%" if pct else f"{v:,.{nd}f}"


def table_single(run_dir: Path) -> str:
    rows = load(run_dir)
    hdr = ["scenario", "conc", "out tok/s", "out tok/s/GPU", "total tok/s", "p50 TTFT ms",
           "p99 TTFT ms", "p50 TPOT ms", "p99 TPOT ms", "accept len", "cache hit", "runs (spread)"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for (sc, c), r in sorted(rows.items()):
        lines.append("| " + " | ".join([
            sc, str(c), fmt(r["output_throughput"]), fmt(r["output_throughput"] / NUM_GPUS),
            fmt(r["total_token_throughput"]), fmt(r["median_ttft_ms"]), fmt(r["p99_ttft_ms"]),
            fmt(r["median_tpot_ms"]), fmt(r["p99_tpot_ms"]), fmt(r.get("accept_len"), 2),
            fmt(r.get("hit_rate"), pct=True),
            f"{r['n']}" + (f" (±{100 * r['spread'] / 2:.1f}%)" if r["spread"] is not None else ""),
        ]) + " |")
    return f"### {run_dir.name}\n\n" + "\n".join(lines)


def table_compare(base_dir: Path, new_dir: Path) -> str:
    base, new = load(base_dir), load(new_dir)
    keys = sorted(set(base) & set(new))
    hdr = ["scenario", "conc", "base out tok/s", "new out tok/s", "speedup",
           "base p50 TPOT", "new p50 TPOT", "base p50 TTFT", "new p50 TTFT"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for k in keys:
        b, n = base[k], new[k]
        sp = n["output_throughput"] / b["output_throughput"]
        lines.append("| " + " | ".join([
            k[0], str(k[1]), fmt(b["output_throughput"]), fmt(n["output_throughput"]), f"{sp:.2f}x",
            fmt(b["median_tpot_ms"]), fmt(n["median_tpot_ms"]),
            fmt(b["median_ttft_ms"]), fmt(n["median_ttft_ms"]),
        ]) + " |")
    # Peak-vs-peak per scenario: each side at its own best concurrency.
    peak = ["", "**Peak output throughput per scenario (each side at its best concurrency):**", ""]
    for sc in sorted({k[0] for k in base} | {k[0] for k in new}):
        pb = max((r["output_throughput"] for (s, _), r in base.items() if s == sc), default=0)
        pn = max((r["output_throughput"] for (s, _), r in new.items() if s == sc), default=0)
        if pb and pn:
            peak.append(f"- {sc}: {pb:,.0f} -> {pn:,.0f} tok/s ({pn / pb:.2f}x)")
    return f"### {base_dir.name} -> {new_dir.name}\n\n" + "\n".join(lines + peak)


def main(argv: list[str]) -> None:
    if len(argv) == 2:
        out = table_single(Path(argv[1]))
        target = Path(argv[1]) / "summary.md"
    elif len(argv) == 3:
        out = table_compare(Path(argv[1]), Path(argv[2]))
        target = Path(argv[2]) / f"compare_vs_{Path(argv[1]).name}.md"
    else:
        sys.exit(__doc__)
    print(out)
    target.write_text(out + "\n")


if __name__ == "__main__":
    main(sys.argv)
