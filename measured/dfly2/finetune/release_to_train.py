# Released DFly2 (AngelSlim/DS-V4.1-Flash-DFly2-B6, V4.1-native names, FP4 experts) -> AngelSpec
# training layout (HF dir for training.draft_init_path), plus a draft config matching the release.
# usage: release_to_train.py <release dir> <out dir>
import json, os, sys
import torch
from safetensors.torch import save_file
sys.path[:0] = ["/data/nvme3/dfly2/pkg", "/data/nvme3/dfly2/AngelSpec"]
from dfly2_v41.ref import DRAFT_CFG, checkpoint_tensors, map_name
from angelspec.models.target.block_quant import dequant_fp4_packed
from safetensors import safe_open

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
cfg = json.load(open(DRAFT_CFG))
# Serving-time choices the release was trained with (from its dfly_config; same as ref.py).
cfg.update(hidden_correction_intermediate_size=5120, attentive_correction=True,
           attentive_correction_source_layer=-1, attentive_correction_block_keys=True,
           attentive_correction_block_mode="gated", attentive_correction_block_kv_in="hidden_embed",
           enable_confidence_head=False)
json.dump(cfg, open(f"{out}/config.json", "w"), indent=2)
json.dump(cfg, open(f"{out}/../draft_config_ft.json", "w"), indent=2)
state, experts, skipped, handles = {}, {}, {}, {}
for name, (fname, _) in checkpoint_tensors(src).items():
    mapped, kind = map_name(name)
    if mapped is None:
        skipped[kind] = skipped.get(kind, 0) + 1; continue
    if fname not in handles:
        handles[fname] = safe_open(os.path.join(src, fname), "pt", device="cpu")
    t = handles[fname].get_tensor(name)
    if isinstance(kind, tuple):
        experts.setdefault(mapped, {}).setdefault(kind[1], {})[kind[2]] = t
    else:
        state[mapped] = t.contiguous()
for mapped, by_e in experts.items():
    n = max(by_e) + 1
    ws = [dequant_fp4_packed(by_e[e]["weight"], by_e[e]["scale"].float()).t().to(torch.bfloat16) for e in range(n)]
    state[mapped] = torch.stack(ws).contiguous()  # [experts, in, out]
save_file(state, f"{out}/model.safetensors")
print(f"tensors written {len(state)} (experts stacked: {len(experts)}), skipped {skipped}")
