"""Re-block MXFP8 (FP8-E4M3 + one UE8M0 scale per 1x32) weights into the
128x128 block-FP8 layout that DeepGEMM's SM90 (Hopper) FP8 GEMM consumes.

DeepSeek-V4.1 ships its dense linears as FP8 with 32x32 power-of-two (UE8M0)
block scales. Hopper has no native kernel for 32-wide scale groups, so vLLM
dequantizes them to BF16 at load and runs cuBLAS BF16. That doubles the
weight bytes and halves tensor-core throughput versus FP8.

Because every scale is a power of two, re-blocking is just exponent shifts.
For each 128x128 tile we take the largest sub-block exponent E, keep one
FP32 scale 2^(E-127) for the tile, and multiply each element by
2^(e_sub - E) <= 1. This is exact as long as the shifted value stays a
normal E4M3 number (|x| >= 2^-6). Only elements pushed into the subnormal
range lose mantissa bits. `reblock_mxfp8_to_block128` reports that loss so
the caller can refuse a layer whose error is too high.

Pure PyTorch; runs on CPU or GPU. The unit tests run on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MX_GROUP = 32
BLOCK = 128
E8M0_BIAS = 127


@dataclass
class ReblockStats:
    rel_err: float          # ||W' - W||_F / ||W||_F over the dequantized weights
    max_shift: int          # largest exponent shift applied to any 1x32 group
    frac_changed: float     # fraction of non-zero FP8 codes whose value changed
    numel: int
    # Worst relative error of any single 1x32 group with a non-zero reference.
    # The whole-matrix ratio can hide a low-norm group crushed to zero; this can't.
    max_group_rel_err: float = 0.0


def _codes_to_int(scale: torch.Tensor) -> torch.Tensor:
    """UE8M0 scales (uint8 or float8_e8m0fnu) -> int32 exponent codes."""
    if scale.dtype == torch.uint8:
        return scale.to(torch.int32)
    if hasattr(torch, "float8_e8m0fnu") and scale.dtype == torch.float8_e8m0fnu:
        return scale.view(torch.uint8).to(torch.int32)
    raise TypeError(f"expected UE8M0 scales (uint8/float8_e8m0fnu), got {scale.dtype}")


def dequant_mxfp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reference dequant: weight [N, K] fp8, scale [N, K/32] UE8M0 -> fp32."""
    n, k = weight.shape
    codes = _codes_to_int(scale)[:n, : k // MX_GROUP]
    s = torch.exp2((codes - E8M0_BIAS).to(torch.float32))
    return (weight.to(torch.float32).view(n, k // MX_GROUP, MX_GROUP) * s.unsqueeze(-1)).view(n, k)


def dequant_block128(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reference dequant: weight [N, K] fp8, scale [ceil(N/128), K/128] fp32 -> fp32."""
    n, k = weight.shape
    s = scale.repeat_interleave(BLOCK, 0)[:n].repeat_interleave(BLOCK, 1)[:, :k]
    return weight.to(torch.float32) * s


@torch.no_grad()
def reblock_mxfp8_to_block128(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    row_chunk: int = 4096,
    compute_stats: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, ReblockStats | None]:
    """Convert MXFP8 [N, K] (scales [N, K/32]) to block-FP8 with [N/128, K/128] FP32 scales.

    Args:
        weight: float8_e4m3fn [N, K], K a multiple of 128.
        scale: UE8M0 codes [>=N, >=K/32] (padding beyond N, K/32 is ignored).
        row_chunk: rows processed at a time (multiple of 128) to bound memory.
        compute_stats: also measure the re-blocking error.

    Returns:
        (new_weight fp8 [N, K], new_scale fp32 [ceil(N/128), K/128], stats)
    """
    assert weight.dtype == torch.float8_e4m3fn and weight.dim() == 2, weight.dtype
    n, k = weight.shape
    if k % BLOCK:
        raise ValueError(f"K={k} must be a multiple of {BLOCK}")
    assert row_chunk % BLOCK == 0
    kg = k // MX_GROUP            # 1x32 groups per row
    kb = k // BLOCK               # 128-wide blocks per row
    nb = (n + BLOCK - 1) // BLOCK
    codes = _codes_to_int(scale)[:n, :kg]
    if codes.shape != (n, kg):
        raise ValueError(f"scale shape {tuple(scale.shape)} too small for weight {(n, k)}")

    # Per-tile max exponent. Pad rows with code 0 (2^-127), which never wins a max.
    pad_rows = nb * BLOCK - n
    codes_p = torch.nn.functional.pad(codes, (0, 0, 0, pad_rows)) if pad_rows else codes
    tile_e = codes_p.view(nb, BLOCK, kb, BLOCK // MX_GROUP).amax(dim=(1, 3))  # [nb, kb]
    new_scale = torch.exp2((tile_e - E8M0_BIAS).to(torch.float32))

    new_weight = torch.empty_like(weight)
    err_sq = 0.0
    ref_sq = 0.0
    changed = 0
    nonzero = 0
    max_shift = 0
    max_group = 0.0
    for r0 in range(0, n, row_chunk):
        r1 = min(n, r0 + row_chunk)
        rows = r1 - r0
        c = codes[r0:r1]                                            # [rows, kg]
        e = tile_e[r0 // BLOCK : (r1 + BLOCK - 1) // BLOCK]         # [rows/128.., kb]
        e = e.repeat_interleave(BLOCK, 0)[:rows].repeat_interleave(BLOCK // MX_GROUP, 1)
        shift = (e - c)                                             # >= 0, [rows, kg]
        if shift.numel():
            max_shift = max(max_shift, int(shift.max()))
        w = weight[r0:r1].to(torch.float32).view(rows, kg, MX_GROUP)
        w_shifted = w * torch.exp2(-shift.to(torch.float32)).unsqueeze(-1)
        q = w_shifted.view(rows, k).to(torch.float8_e4m3fn)
        new_weight[r0:r1] = q
        if compute_stats:
            # Compare in the original scale domain.
            tile_scale = torch.exp2((e - E8M0_BIAS).to(torch.float32))    # [rows, kg]
            orig_scale = torch.exp2((c - E8M0_BIAS).to(torch.float32))
            ref = w * orig_scale.unsqueeze(-1)
            new = q.to(torch.float32).view(rows, kg, MX_GROUP) * tile_scale.unsqueeze(-1)
            g_err = ((new - ref) ** 2).sum(-1)                         # [rows, kg]
            g_ref = (ref ** 2).sum(-1)
            err_sq += float(g_err.sum())
            ref_sq += float(g_ref.sum())
            live = g_ref > 0
            if bool(live.any()):
                max_group = max(max_group, float((g_err[live] / g_ref[live]).max().sqrt()))
            nz = w != 0
            nonzero += int(nz.sum())
            changed += int(((new != ref) & nz).sum())

    stats = None
    if compute_stats:
        stats = ReblockStats(
            rel_err=(err_sq / ref_sq) ** 0.5 if ref_sq > 0 else 0.0,
            max_shift=max_shift,
            frac_changed=changed / nonzero if nonzero else 0.0,
            numel=n * k,
            max_group_rel_err=max_group,
        )
    return new_weight, new_scale, stats
