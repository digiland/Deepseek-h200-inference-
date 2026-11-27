"""Reference (oracle) DFly2 drafter for DeepSeek-V4.1-Flash, built from AngelSpec's own modules.

Loads the released checkpoint (AngelSlim/DS-V4.1-Flash-DFly2-B6, V4.1-native tensor names) into
AngelSpec's ``DFly2DraftModel`` so that serving code can be checked against the training code.
"""

import json
import os
import re
import struct
from typing import Dict, Iterable, Tuple

import torch

ANGELSPEC = "/data/nvme3/dfly2/AngelSpec"
DRAFT_CFG = f"{ANGELSPEC}/angelspec/config/dfly2_dsv41_flash_draft_config.json"


def angelspec_config():
    from angelspec.models.draft.dfly2 import DFly2Config

    with open(DRAFT_CFG) as f:
        cfg = json.load(f)
    # Serving-time choices recorded in the released checkpoint's dfly_config.
    cfg.update(
        hidden_correction_intermediate_size=5120,
        attentive_correction=True,
        attentive_correction_source_layer=-1,
        attentive_correction_block_keys=True,
        attentive_correction_block_mode="gated",
        attentive_correction_block_kv_in="hidden_embed",
        dsa_backend="torch",
        enable_confidence_head=False,
    )
    for key in ("architectures", "model_type"):
        cfg.pop(key, None)
    return DFly2Config(**cfg)


def read_header(path: str) -> Dict[str, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")
_W2PROJ = {"w1": "experts_gate_proj", "w3": "experts_up_proj", "w2": "experts_down_proj"}
_SHARED = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}


def map_name(name: str):
    """Released name -> (AngelSpec param name, kind) or (None, reason)."""
    if not name.startswith("model."):
        return None, "unexpected prefix"
    rest = name[len("model."):]
    m = EXPERT_RE.match(name)
    if m:
        layer, expert, w, part = m.groups()
        return f"layers.{layer}.mlp.{_W2PROJ[w]}.weight", ("expert", int(expert), part)
    m = re.match(r"^layers\.(\d+)\.(.*)$", rest)
    if m:
        layer, tail = m.groups()
        p = f"layers.{layer}."
        if tail.startswith("attn.") :
            return p + "self_attn." + tail[len("attn."):], "plain"
        if tail == "attn_norm.weight":
            return p + "input_layernorm.weight", "plain"
        if tail == "ffn_norm.weight":
            return p + "post_attention_layernorm.weight", "plain"
        if tail.startswith("hc_"):
            return p + tail, "plain"
        if tail == "ffn.gate.weight":
            return p + "mlp.router.gate.weight", "plain"
        if tail == "ffn.gate.bias":
            return p + "mlp.expert_bias", "plain"
        if tail == "ffn.gate.bias_vl":
            return None, "image-token router bias (unused for text)"
        m2 = re.match(r"^ffn\.shared_experts\.(w[123])\.weight$", tail)
        if m2:
            return p + f"mlp.shared_mlp.{_SHARED[m2.group(1)]}.weight", "plain"
        return None, "unmapped layer tensor"
    if rest.startswith(("final_norm.", "hidden_correction.", "mtp_fusion.")):
        return rest, "plain"
    return None, "unmapped"


def checkpoint_tensors(ckpt: str) -> Dict[str, Tuple[str, dict]]:
    out = {}
    for f in sorted(os.listdir(ckpt)):
        if f.endswith(".safetensors"):
            for k, v in read_header(os.path.join(ckpt, f)).items():
                out[k] = (f, v)
    return out


def build_reference(ckpt: str, target: str, device="cuda", dtype=torch.bfloat16):
    """AngelSpec DFly2DraftModel with the released weights (FP4 experts dequantized), plus the
    target's embedding and LM head. Returns (model, lm_head_weight)."""
    from safetensors import safe_open

    from angelspec.models.draft.dfly2 import DFly2DraftModel
    from angelspec.models.target.block_quant import dequant_fp4_packed

    cfg = angelspec_config()
    with torch.device("meta"):
        shapes = {k: tuple(v.shape) for k, v in DFly2DraftModel(cfg).named_parameters()}
    model = DFly2DraftModel(cfg)  # real buffers (RoPE tables etc.); weights replaced below
    params = {k: torch.empty(0).new_empty(v) for k, v in shapes.items()}
    state: Dict[str, torch.Tensor] = {}
    experts: Dict[Tuple[str, int], Dict[str, torch.Tensor]] = {}
    tensors = checkpoint_tensors(ckpt)
    handles = {}
    for name, (fname, _) in tensors.items():
        mapped, kind = map_name(name)
        if mapped is None:
            continue
        if fname not in handles:
            handles[fname] = safe_open(os.path.join(ckpt, fname), "pt", device="cpu")
        t = handles[fname].get_tensor(name)
        if isinstance(kind, tuple):
            _, e, part = kind
            experts.setdefault((mapped, e), {})[part] = t
        else:
            state[mapped] = t
    for (mapped, e), parts in experts.items():
        w = dequant_fp4_packed(parts["weight"], parts["scale"].float()).t()  # [in, out]
        if mapped not in state:
            p = params[mapped]
            state[mapped] = torch.empty(p.shape, dtype=dtype)
        state[mapped][e].copy_(w.to(dtype))
    with open(os.path.join(target, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    with safe_open(os.path.join(target, wm["embed.weight"]), "pt", device="cpu") as f:
        state["embed_tokens.weight"] = f.get_tensor("embed.weight")
    with safe_open(os.path.join(target, wm["head.weight"]), "pt", device="cpu") as f:
        lm_head = f.get_tensor("head.weight")
    fp32 = {"mlp.expert_bias", "mlp.router.gate.weight", "attn_sink"}
    for k in list(state):
        keep32 = any(s in k for s in fp32) or ".hc_" in k or k.split(".")[-1].startswith("hc_")
        state[k] = state[k].to(torch.float32 if keep32 else dtype)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    model = model.to(device).eval()
    return model, lm_head.to(device=device, dtype=dtype), missing, unexpected
