"""CPU tests for the MXFP8 -> block-128 FP8 re-blocking (no GPU needed)."""

import pytest
import torch

from h200opt.reblock import (
    dequant_block128,
    dequant_mxfp8,
    reblock_mxfp8_to_block128,
)


def _make_mxfp8(n, k, code_lo, code_hi, seed=0, row_uniform32=True):
    """Random FP8 weight plus UE8M0 codes, like the checkpoint (32x32 blocks)."""
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
    if row_uniform32:
        blk = torch.randint(code_lo, code_hi + 1, ((n + 31) // 32, k // 32), generator=g)
        codes = blk.repeat_interleave(32, 0)[:n]
    else:
        codes = torch.randint(code_lo, code_hi + 1, (n, k // 32), generator=g)
    return w, codes.to(torch.uint8)


def test_uniform_scales_are_bit_exact():
    w, codes = _make_mxfp8(256, 512, 120, 120)
    q, s, st = reblock_mxfp8_to_block128(w, codes)
    assert torch.equal(q.view(torch.uint8), w.view(torch.uint8))
    assert st.rel_err == 0.0 and st.max_shift == 0
    assert s.shape == (2, 4)
    assert torch.allclose(s, torch.full_like(s, 2.0 ** (120 - 127)))


def test_small_shifts_are_exact():
    # Shifts of at most 2 keep every |x| >= 2^-4 normal; only tiny values may round.
    w, codes = _make_mxfp8(384, 1280, 118, 120, seed=1)
    q, s, st = reblock_mxfp8_to_block128(w, codes)
    ref = dequant_mxfp8(w, codes)
    new = dequant_block128(q, s)
    assert st.max_shift <= 2
    assert st.rel_err < 1e-3, st
    assert torch.allclose(new, ref, rtol=0.13, atol=float(ref.abs().max()) * 2**-8)


@pytest.mark.parametrize("n,k", [(1792, 5120), (100, 256), (4096, 1280)])
def test_shapes_and_padding(n, k):
    w, codes = _make_mxfp8(n, k, 110, 125, seed=2)
    q, s, st = reblock_mxfp8_to_block128(w, codes, row_chunk=256)
    assert q.shape == (n, k) and q.dtype == torch.float8_e4m3fn
    assert s.shape == ((n + 127) // 128, k // 128) and s.dtype == torch.float32
    # stats agree with an independent dequant comparison
    ref = dequant_mxfp8(w, codes)
    new = dequant_block128(q, s)
    rel = float((new - ref).norm() / ref.norm())
    assert abs(rel - st.rel_err) < 1e-6 + 1e-3 * rel


def test_padded_scale_tensor_is_sliced():
    w, codes = _make_mxfp8(128, 256, 119, 121)
    padded = torch.nn.functional.pad(codes, (0, 4, 0, 8), value=200)
    q1, s1, _ = reblock_mxfp8_to_block128(w, codes)
    q2, s2, _ = reblock_mxfp8_to_block128(w, padded)
    assert torch.equal(q1.view(torch.uint8), q2.view(torch.uint8)) and torch.equal(s1, s2)


def test_large_spread_reports_error():
    # A tile mixing scales 2^-20 and 2^0 crushes the small block into subnormals.
    w, codes = _make_mxfp8(128, 128, 107, 127, seed=3, row_uniform32=False)
    _, _, st = reblock_mxfp8_to_block128(w, codes)
    assert st.max_shift >= 10
    assert st.frac_changed > 0


def test_e8m0_dtype_accepted():
    w, codes = _make_mxfp8(128, 256, 120, 122)
    e8 = codes.view(torch.float8_e8m0fnu)
    q1, s1, _ = reblock_mxfp8_to_block128(w, codes)
    q2, s2, _ = reblock_mxfp8_to_block128(w, e8)
    assert torch.equal(q1.view(torch.uint8), q2.view(torch.uint8)) and torch.equal(s1, s2)


def test_rejects_bad_k():
    w = torch.zeros(128, 96).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError):
        reblock_mxfp8_to_block128(w, torch.zeros(128, 3, dtype=torch.uint8))
