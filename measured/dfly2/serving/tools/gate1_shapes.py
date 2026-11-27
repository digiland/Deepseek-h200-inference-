"""Gate 1: map every released DFly2 tensor onto AngelSpec's DFly2DraftModel (meta device) and
check shapes, including FP4 expert unpacking. CPU only."""

import collections
import sys

import torch

sys.path.insert(0, "/data/nvme3/dfly2/pkg")
from dfly2_v41.ref import angelspec_config, checkpoint_tensors, map_name  # noqa: E402

CKPT = "/data/nvme3/dfly2/DS-V4.1-Flash-DFly2-B6"


def main():
    from angelspec.models.draft.dfly2 import DFly2DraftModel
    from angelspec.models.target.block_quant import dequant_fp4_packed
    from safetensors import safe_open

    cfg = angelspec_config()
    with torch.device("meta"):
        model = DFly2DraftModel(cfg)
    params = {k: tuple(v.shape) for k, v in model.named_parameters()}
    tensors = checkpoint_tensors(CKPT)
    print(f"checkpoint tensors: {len(tensors)}; AngelSpec parameters: {len(params)}")

    filled = collections.Counter()
    skipped = collections.Counter()
    bad = []
    expert_seen = collections.defaultdict(set)
    for name, (_, meta) in tensors.items():
        mapped, kind = map_name(name)
        if mapped is None:
            skipped[kind] += 1
            continue
        if mapped not in params:
            bad.append(f"{name} -> {mapped}: no such AngelSpec parameter")
            continue
        shape = tuple(meta["shape"])
        if isinstance(kind, tuple):
            _, e, part = kind
            E, din, dout = params[mapped]
            if part == "weight":
                want = (dout, din // 2)  # [out, in/2] packed int8
                if shape != want:
                    bad.append(f"{name}: {shape} vs packed {want}")
                expert_seen[mapped].add(e)
            else:
                want = (dout, din // 32)
                if shape != want:
                    bad.append(f"{name}: scale {shape} vs {want}")
            continue
        if shape != params[mapped]:
            bad.append(f"{name}: {shape} vs AngelSpec {mapped} {params[mapped]}")
        filled[mapped] += 1

    expert_params = {k for k in params if ".experts_" in k}
    for k in expert_params:
        n = len(expert_seen.get(k, ()))
        if n != params[k][0]:
            bad.append(f"{k}: {n} of {params[k][0]} experts in checkpoint")
    shared_from_target = {"embed_tokens.weight"}
    unfilled = sorted(set(params) - set(filled) - expert_params - shared_from_target)

    # Unpack one FP4 expert to check the format decodes to finite, sane values.
    name = "model.layers.0.ffn.experts.0.w1.weight"
    f = tensors[name][0]
    with safe_open(f"{CKPT}/{f}", "pt") as h:
        w = h.get_tensor(name)
        s = h.get_tensor(name.replace(".weight", ".scale"))
    d = dequant_fp4_packed(w, s.float())
    print(f"FP4 expert 0 w1: packed {tuple(w.shape)} {w.dtype}, scale {tuple(s.shape)} {s.dtype}"
          f" -> {tuple(d.shape)}, finite={bool(torch.isfinite(d).all())}, rms={d.pow(2).mean().sqrt():.4g}")

    print(f"mapped non-expert tensors: {sum(filled.values())}; expert projections: "
          f"{len(expert_params)} x {params[next(iter(expert_params))][0]} experts")
    print("skipped:", dict(skipped))
    print("AngelSpec params left unfilled (besides target embedding):", unfilled)
    print("shape mismatches / errors:", len(bad))
    for b in bad[:30]:
        print("  ", b)
    ok = not bad and not unfilled
    print("GATE1", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
