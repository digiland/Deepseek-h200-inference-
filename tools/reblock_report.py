#!/usr/bin/env python3
"""Measure, on the real checkpoint, how lossy MXFP8 -> 128x128 block-FP8 re-blocking is.

CPU only. Reads the dense FP8 linears (skips routed experts and Engram tables)
straight from the safetensors shards, and reports the error per tensor plus a
summary. Run this once on the node before enabling the FP8 dense path:

    python tools/reblock_report.py /path/to/DeepSeek-V4.1-Flash-1017 --out results/reblock.csv

The plugin's per-layer fallback threshold is H200OPT_MXFP8_MAX_REL_ERR (default 5e-3).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "h200opt"))
from h200opt.reblock import reblock_mxfp8_to_block128  # noqa: E402

SCALE_SUFFIXES = (".scale", ".weight_scale", ".weight_scale_inv")
SKIP = re.compile(r"\.experts\.|engram\.embed|\.embed\b|tok_embeddings|vision\.|aligner\.")


def expand_rows(scale: torch.Tensor, n: int, k: int) -> torch.Tensor | None:
    """32x32 block scales [ceil(N/32), K/32] -> per-row [N, K/32]; per-row scales pass through."""
    kg = k // 32
    if scale.shape == (n, kg):
        return scale
    if scale.shape[0] == (n + 31) // 32 and scale.shape[1] == kg:
        return scale.repeat_interleave(32, 0)[:n]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--out", default=None, help="CSV with one row per tensor")
    ap.add_argument("--limit", type=int, default=0, help="stop after N tensors (0 = all)")
    ap.add_argument("--threshold", type=float, default=5e-3, help="whole-matrix limit")
    ap.add_argument("--group-threshold", type=float, default=0.1, help="worst 1x32 group limit")
    args = ap.parse_args()

    from safetensors import safe_open

    root = Path(args.model_dir)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    by_file: dict[str, list[str]] = {}
    for name, f in index.items():
        by_file.setdefault(f, []).append(name)

    rows = []
    done = 0
    for fname in sorted(by_file):
        names = set(by_file[fname])
        with safe_open(str(root / fname), framework="pt") as fh:
            for name in sorted(names):
                if not name.endswith(".weight") or SKIP.search(name):
                    continue
                base = name[: -len(".weight")]
                sname = next((base + s for s in SCALE_SUFFIXES if base + s in index), None)
                if sname is None:
                    continue
                w = fh.get_tensor(name)
                if w.dtype != torch.float8_e4m3fn or w.dim() != 2 or w.shape[1] % 128:
                    continue
                if index[sname] == fname:
                    s = fh.get_tensor(sname)
                else:
                    with safe_open(str(root / index[sname]), framework="pt") as fs:
                        s = fs.get_tensor(sname)
                if s.dtype not in (torch.uint8, getattr(torch, "float8_e8m0fnu", None)):
                    continue
                s = expand_rows(s, *w.shape)
                if s is None:
                    continue
                _, _, st = reblock_mxfp8_to_block128(w, s)
                rows.append({"tensor": name, "N": w.shape[0], "K": w.shape[1],
                             "rel_err": st.rel_err, "max_group_rel_err": st.max_group_rel_err,
                             "max_shift": st.max_shift,
                             "frac_changed": st.frac_changed})
                done += 1
                bad = st.rel_err > args.threshold or st.max_group_rel_err > args.group_threshold
                flag = "  <-- would stay BF16" if bad else ""
                print(f"{name:70s} {tuple(w.shape)!s:>14} rel_err={st.rel_err:.2e} "
                      f"group<={st.max_group_rel_err:.2e} "
                      f"shift<={st.max_shift:2d} changed={st.frac_changed:.2%}{flag}", flush=True)
                if args.limit and done >= args.limit:
                    break
        if args.limit and done >= args.limit:
            break

    if not rows:
        sys.exit("no MXFP8 dense tensors found; check the checkpoint path / naming")
    worst = max(rows, key=lambda r: r["rel_err"])
    over = sum(r["rel_err"] > args.threshold or r["max_group_rel_err"] > args.group_threshold
               for r in rows)
    params = sum(r["N"] * r["K"] for r in rows)
    print(f"\n{len(rows)} tensors, {params/1e9:.2f}B params; worst rel_err {worst['rel_err']:.2e} "
          f"({worst['tensor']}); {over} over the limits (those stay BF16).\n"
          "Note: this measures weight re-blocking only. The FP8 path also quantizes "
          "activations, so check output quality with bench/quality_gate.py.")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)


if __name__ == "__main__":
    main()
