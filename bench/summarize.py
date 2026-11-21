#!/usr/bin/env python3
"""Summarize `vllm bench serve` result JSONs, and optionally compare two runs.

    python bench/summarize.py results/vllm-baseline
    python bench/summarize.py results/vllm-baseline results/vllm-opt   # speedup table

Throughput numbers are reported both absolute and per GPU (NUM_GPUS env, default 4).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

NUM_GPUS = int(os.environ.get("NUM_GPUS", "4"))
FIELDS = [
    ("output_throughput", "out tok/s"),
    ("total_token_throughput", "total tok/s"),
    ("median_ttft_ms", "p50 TTFT ms"),
    ("p99_ttft_ms", "p99 TTFT ms"),
    ("median_tpot_ms", "p50 TPOT ms"),
    ("p99_tpot_ms", "p99 TPOT ms"),
]
NAME_RE = re.compile(r"(?P<sc>[a-z]+)_c(?P<c>\d+)\.json$")


def load(run_dir: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    for f in sorted(run_dir.glob("*_c*.json")):
        m = NAME_RE.search(f.name)
        if not m:
            continue
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        failed = d.get("failed", 0) or 0
        rows[(m["sc"], int(m["c"]))] = {
            **{k: d.get(k) for k, _ in FIELDS},
            "completed": d.get("completed"),
            "failed": failed,
        }
    return rows


def server_counters(run_dir: Path) -> str:
    """Prefix-cache hit rate and spec-decode acceptance from the scraped /metrics
    (cumulative over the whole server lifetime, so warmups are included)."""
    f = run_dir / "server_metrics.prom"
    if not f.exists():
        return ""
    tot: dict[str, float] = {}
    for line in f.read_text().splitlines():
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
    out = []
    q = tot.get("vllm:prefix_cache_queries_total", tot.get("vllm:prefix_cache_queries"))
    h = tot.get("vllm:prefix_cache_hits_total", tot.get("vllm:prefix_cache_hits"))
    if q:
        out.append(f"prefix-cache hit rate {100 * (h or 0) / q:.1f}% ({h:,.0f}/{q:,.0f} tokens)")
    drafts = tot.get("vllm:spec_decode_num_drafts_total", tot.get("vllm:spec_decode_num_drafts"))
    acc = tot.get("vllm:spec_decode_num_accepted_tokens_total",
                  tot.get("vllm:spec_decode_num_accepted_tokens"))
    dtok = tot.get("vllm:spec_decode_num_draft_tokens_total",
                   tot.get("vllm:spec_decode_num_draft_tokens"))
    if drafts:
        out.append(f"DSpark mean acceptance length {1 + (acc or 0) / drafts:.2f} tokens/step"
                   + (f", draft acceptance {100 * acc / dtok:.1f}%" if dtok else ""))
    for k in ("sglang:spec_accept_length", "sglang:cache_hit_rate"):
        if k in tot:
            out.append(f"{k} = {tot[k]:.3f}")
    return ("\n\nServer counters: " + "; ".join(out)) if out else ""


def fmt(v, nd=1):
    return "-" if v is None else f"{v:,.{nd}f}"


def table_single(run_dir: Path) -> str:
    rows = load(run_dir)
    hdr = ["scenario", "conc"] + [h for _, h in FIELDS] + ["out tok/s/GPU", "ok/fail"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for (sc, c), r in sorted(rows.items()):
        per_gpu = None if r["output_throughput"] is None else r["output_throughput"] / NUM_GPUS
        lines.append(
            "| "
            + " | ".join(
                [sc, str(c)]
                + [fmt(r[k]) for k, _ in FIELDS]
                + [fmt(per_gpu), f"{r['completed']}/{r['failed']}"]
            )
            + " |"
        )
    return f"### {run_dir.name}\n\n" + "\n".join(lines) + server_counters(run_dir)


def table_compare(base_dir: Path, new_dir: Path) -> str:
    base, new = load(base_dir), load(new_dir)
    keys = sorted(set(base) & set(new))
    hdr = ["scenario", "conc", "base out tok/s", "new out tok/s", "speedup",
           "base p50 TPOT", "new p50 TPOT", "base p50 TTFT", "new p50 TTFT"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    best = {}
    for k in keys:
        b, n = base[k], new[k]
        sp = (n["output_throughput"] / b["output_throughput"]
              if b["output_throughput"] and n["output_throughput"] else None)
        lines.append("| " + " | ".join([
            k[0], str(k[1]), fmt(b["output_throughput"]), fmt(n["output_throughput"]),
            "-" if sp is None else f"{sp:.2f}x",
            fmt(b["median_tpot_ms"]), fmt(n["median_tpot_ms"]),
            fmt(b["median_ttft_ms"]), fmt(n["median_ttft_ms"]),
        ]) + " |")
    # Peak-vs-peak per scenario: the fairest "throughput doubling" metric,
    # because each engine is allowed to pick its own best concurrency.
    for label, rows in (("base", base), ("new", new)):
        for (sc, _), r in rows.items():
            v = r["output_throughput"] or 0
            best.setdefault(sc, {}).setdefault(label, 0)
            best[sc][label] = max(best[sc][label], v)
    peak = ["", "**Peak output throughput per scenario (each side at its best concurrency):**", ""]
    for sc, d in sorted(best.items()):
        if d.get("base"):
            peak.append(f"- {sc}: {d['base']:,.0f} -> {d.get('new', 0):,.0f} tok/s "
                        f"({d.get('new', 0) / d['base']:.2f}x)")
    return f"### {base_dir.name} -> {new_dir.name}\n\n" + "\n".join(lines + peak)


def main(argv: list[str]) -> None:
    if len(argv) == 2:
        out = table_single(Path(argv[1]))
    elif len(argv) == 3:
        out = table_compare(Path(argv[1]), Path(argv[2]))
    else:
        sys.exit(__doc__)
    print(out)
    target = Path(argv[-1]) / ("summary.md" if len(argv) == 2 else f"compare_vs_{Path(argv[1]).name}.md")
    target.write_text(out + "\n")


if __name__ == "__main__":
    main(sys.argv)
