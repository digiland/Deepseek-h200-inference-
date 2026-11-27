# AngelSpec training-layout DFly2 weights -> released serving layout (V4.1-native names, MXFP4
# experts with E8M0 scales per 32 inputs), so the fine-tuned drafter serves like the release.
# usage: train_to_release.py <train-layout dir with model.safetensors> <release dir (template)> <out dir>
import json, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file
sys.path[:0] = ["/data/nvme3/dfly2/pkg", "/data/nvme3/dfly2/AngelSpec"]
from dfly2_v41.ref import checkpoint_tensors, map_name
from angelspec.models.target.block_quant import FP4_E2M1_VALUES

def load_dir(d):
    out = {}
    if os.path.isfile(os.path.join(d, "pytorch_model.bin")):   # trainer's hf_final
        sd = torch.load(os.path.join(d, "pytorch_model.bin"), map_location="cpu", weights_only=True, mmap=True)
        return {k: v for k, v in sd.items()}
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(d, f), "pt", device="cpu") as h:
                for k in h.keys(): out[k] = h.get_tensor(k)
    return out

_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

def quant_mxfp4(w, block=32):
    """[out, in] float -> (int8 [out, in/2] two e2m1 per byte, low nibble first; e8m0 [out, in/32])."""
    w = w.float()
    o, i = w.shape
    g = w.view(o, i // block, block)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(2.0 ** -126)
    exp = torch.ceil(torch.log2(amax / 6.0))           # smallest power of two with amax/scale <= 6
    scale = torch.exp2(exp)
    x = (g / scale).clamp(-6, 6)
    mag = x.abs().unsqueeze(-1)
    idx = (mag - _POS).abs().argmin(-1)                  # nearest e2m1 magnitude (ties -> lower)
    code = idx + 8 * ((x < 0) & (idx > 0)).long()        # sign bit; -0 stored as +0
    code = code.view(o, i).to(torch.uint8)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).view(torch.int8)
    return packed, scale.view(o, i // block).to(torch.float8_e8m0fnu)

def main(train_dir, release_dir, out_dir):
    trained = load_dir(train_dir)
    rel = checkpoint_tensors(release_dir)
    out, used = {}, set()
    handles = {}
    for name, (fname, meta) in rel.items():
        mapped, kind = map_name(name)
        if mapped is None:   # image-token router bias: keep the release tensor
            handles.setdefault(fname, safe_open(os.path.join(release_dir, fname), "pt", device="cpu"))
            out[name] = handles[fname].get_tensor(name); continue
        used.add(mapped)
        if isinstance(kind, tuple):
            _, e, part = kind
            if part == "scale": continue
            packed, scale = quant_mxfp4(trained[mapped][e].t().contiguous())
            out[name] = packed
            out[name[: -len("weight")] + "scale"] = scale
        else:
            dt = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}[meta["dtype"]]
            out[name] = trained[mapped].to(dt).contiguous()
    left = sorted(set(trained) - used - {"embed_tokens.weight"})
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(release_dir):   # config, tokenizer, chat template, inference code
        src = os.path.join(release_dir, f)
        if f.endswith(".safetensors") or f == "model.safetensors.index.json": continue
        (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, os.path.join(out_dir, f))
    save_file(out, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    json.dump({"metadata": {}, "weight_map": {k: "model.safetensors" for k in out}},
              open(os.path.join(out_dir, "model.safetensors.index.json"), "w"))
    print(f"wrote {len(out)} tensors; trained tensors not exported: {left}")

if __name__ == "__main__":
    main(*sys.argv[1:4])
