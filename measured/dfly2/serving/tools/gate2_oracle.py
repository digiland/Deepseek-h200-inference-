"""Gate 2: compare the vLLM DFly2 drafter with AngelSpec's DFly2 modules on the same inputs.

Reads the step dumps the vLLM speculator writes (DFLY2_DUMP_DIR, rank 0), rebuilds each
request's context latents from the dequantized target cache records the drafter read, runs
AngelSpec's DFly2DraftModel (backbone) and HiddenStatesCorrection (teacher-forced with the
tokens vLLM drafted) and reports per-tensor cosine similarity and per-slot top-1 agreement.
"""

import glob
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/nvme3/dfly2/pkg")
from dfly2_v41.ref import build_reference  # noqa: E402

CKPT = "/data/nvme3/dfly2/DS-V4.1-Flash-DFly2-B6"
TARGET = "/data/nvme3/lab/teachers/deepseek-v4.1-flash-1017"
BLOCK = 6


def cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


@torch.no_grad()
def main(dump_dir):
    dev = os.environ.get("ORACLE_DEVICE", "cuda")
    model, lm_head, missing, unexpected = build_reference(CKPT, TARGET, device=dev)
    print(f"reference built; missing={len(missing)} unexpected={len(unexpected)}")
    files = sorted(glob.glob(os.path.join(dump_dir, "step*_rank0.pt")))
    stats = {"backbone": [], "kv_block": [], "corrected": [], "top1": [], "n": 0}
    for path in files:
        d = torch.load(path)
        R = d["backbone"].shape[0]
        ids = d["input_ids"].view(R, BLOCK).to(dev)
        pos = d["positions"].view(R, BLOCK).to(dev)
        drafted = d["draft_tokens"].to(dev)
        N = drafted.shape[1]
        for r in range(R):
            nw = int(min(d["anchor_pos"][r].item(), 128))
            ctx_lat, idx = [], []
            for i in range(3):
                win = d[f"window{i}"][r, :nw]
                cnt = int(d["topk_len"][i, r])
                top = d[f"topk{i}"][r, :cnt]
                lat = torch.cat([win, top], 0).unsqueeze(0).to(dev)
                L = lat.shape[1]
                rows = torch.cat([torch.arange(L), L + torch.arange(BLOCK)]).to(dev)
                ctx_lat.append(lat)
                idx.append(rows.view(1, 1, -1).expand(1, BLOCK, -1).to(torch.int32).contiguous())
            # self-check: the block latents vLLM inserted (dequantized) vs recomputed
            attn0 = model.layers[0].self_attn
            x0 = d["layer0.attn_in"].view(R, BLOCK, -1)[r : r + 1].to(dev)
            c0, s0 = attn0._rope(pos[r : r + 1])
            kv_ref = attn0._latent(x0, c0, s0)
            kv_vllm = d["window0"][r, nw : nw + BLOCK].to(dev)
            stats["kv_block"].append(cos(kv_ref, kv_vllm))
            # backbone
            emb = model.embed_tokens(ids[r : r + 1])
            delta = model.mtp_fusion(emb[:, 0], d["prev_hidden"][r : r + 1].to(dev), BLOCK)
            noise = emb + delta.view(1, BLOCK, -1)
            h = model(
                draft_input_ids=None,
                context_feature=None,
                draft_position_ids=pos[r : r + 1],
                context_position_ids=pos[r : r + 1],
                noise_embedding=noise,
                sparse_kv_indices=idx,
                context_latents=ctx_lat,
            )
            stats["backbone"].append(cos(h[0], d["backbone"][r].to(dev)))
            # correction, teacher-forced with vLLM's drafted tokens
            prev = torch.cat([ids[r : r + 1, :1], ids[r : r + 1, :1], drafted[r : r + 1, :-1]], 1)
            prev = torch.cat([prev, prev.new_full((1, BLOCK - prev.shape[1]), int(ids[r, 0]))], 1) \
                if prev.shape[1] < BLOCK else prev[:, :BLOCK]
            L2 = ctx_lat[2].shape[1]
            cidx = torch.arange(L2, device=dev).view(1, 1, -1).expand(1, BLOCK, -1)
            context = {"position_ids": pos[r : r + 1], "latent": ctx_lat[2],
                       "indices": cidx.to(torch.int32).contiguous(), "block_size": BLOCK}
            corr = model.hidden_correction(h, model.embed_tokens(prev), context=context)
            for j in range(1, N + 1):
                ref_logits = corr[0, j].float() @ lm_head.float().t()
                v_logits = d[f"logits{j}"][r].to(dev)
                stats["corrected"].append(cos(corr[0, j], d[f"corrected{j}"][r].to(dev)))
                stats["top1"].append(int(ref_logits.argmax() == v_logits.argmax()))
            stats["n"] += 1

    def mean(x):
        return sum(x) / max(len(x), 1)

    print(f"steps={len(files)} requests={stats['n']} slots={len(stats['top1'])}")
    print(f"cosine block KV (vLLM cache vs AngelSpec latent): mean {mean(stats['kv_block']):.5f} "
          f"min {min(stats['kv_block']):.5f}")
    print(f"cosine backbone hidden: mean {mean(stats['backbone']):.5f} "
          f"min {min(stats['backbone']):.5f}")
    print(f"cosine corrected hidden: mean {mean(stats['corrected']):.5f} "
          f"min {min(stats['corrected']):.5f}")
    print(f"top-1 agreement per slot: {mean(stats['top1']):.4f}")
    ok = mean(stats["backbone"]) > 0.99 and mean(stats["top1"]) >= 0.95
    print("GATE2", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1])
