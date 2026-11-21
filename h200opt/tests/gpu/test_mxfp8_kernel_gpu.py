"""GPU test (H100/H200 only): re-blocked FP8 DeepGEMM linear vs the BF16 reference.

Run inside the serving image on one GPU:
    H200OPT_MXFP8_DEEPGEMM=1 python -m pytest -q h200opt/tests/gpu -s
It prints per-shape error and speed so you can see the dense-GEMM win before serving.
"""

import os

import pytest
import torch

os.environ.setdefault("H200OPT_MXFP8_DEEPGEMM", "1")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
    pytest.skip("needs an SM90 GPU", allow_module_level=True)

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402

from h200opt.reblock import dequant_mxfp8  # noqa: E402
from h200opt.vllm_mxfp8_sm90 import H200ReblockMxfp8LinearKernel  # noqa: E402
from vllm.model_executor.kernels.linear.mxfp8 import Mxfp8LinearLayerConfig  # noqa: E402

# DeepSeek-V4.1 dense shapes (N, K): fused wq_a+wkv, wq_b, wo_b, shared w13, shared w2,
# indexer q_b, engram fused kv proj.
SHAPES = [(1792, 5120), (32768, 1280), (5120, 8192), (4608, 5120), (5120, 2304),
          (4096, 1280), (25600, 6144)]
MS = [1, 16, 64, 256, 1536, 8192]


def _mxfp8(n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, device="cuda", generator=g) * 64).clamp(-448, 448)
    w = w.to(torch.float8_e4m3fn)
    # checkpoint-like 32x32 block exponents within a narrow band
    blk = torch.randint(117, 121, ((n + 31) // 32, k // 32), device="cuda", generator=g)
    codes = blk.repeat_interleave(32, 0)[:n].to(torch.uint8)
    return w, codes


class _Layer(torch.nn.Module):
    def __init__(self, w, s):
        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(s, requires_grad=False)


def _bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@pytest.mark.parametrize("n,k", SHAPES)
def test_matches_bf16_reference(n, k):
    with set_current_vllm_config(VllmConfig()):
        ok, why = H200ReblockMxfp8LinearKernel.is_supported()
        assert ok, why
        w, s = _mxfp8(n, k, seed=n + k)
        w_ref = dequant_mxfp8(w, s).to(torch.bfloat16)
        kern = H200ReblockMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
        layer = _Layer(w.clone(), s.clone())
        kern.process_weights_after_loading(layer)
        assert not kern.bf16_fallback
        for m in MS:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            ref = x @ w_ref.t()
            out = kern.apply_weights(layer, x)
            # FP8 activation quantization (1x128) dominates the error; ~2-3% relative.
            rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            t_new = _bench(lambda: kern.apply_weights(layer, x))
            t_ref = _bench(lambda: x @ w_ref.t())
            print(f"N={n:6d} K={k:5d} M={m:5d} rel_err={rel:.4f} "
                  f"fp8={t_new*1e3:8.1f}us bf16={t_ref*1e3:8.1f}us speedup={t_ref/t_new:5.2f}x")
            assert rel < 0.05, rel
