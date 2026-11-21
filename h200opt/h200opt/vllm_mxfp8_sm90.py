"""Native FP8 tensor-core path for MXFP8 dense linears on H100/H200 (SM90).

Stock vLLM dequantizes DeepSeek-V4.1's MXFP8 dense weights to BF16 on SM90
and runs cuBLAS BF16 (plus an FP8 GEMV for <= 24-row decode steps). This
kernel instead re-blocks the weights once at load into 128x128 block-FP8
(see `reblock.py`) and runs DeepGEMM's SM90 FP8 GEMM. Activations are
quantized per token with 1x128 groups, the recipe DeepSeek-V3 serves with
on Hopper.

  * 2x tensor-core peak (FP8 vs BF16) for prefill and large decode batches
  * half the resident bytes of the BF16 copy, which matters most with DP
    attention, where every GPU holds a full replica of the dense weights

Not lossless. Weight re-blocking is exact except for values pushed into FP8
subnormals, but activations also become FP8 (1x128 groups, power-of-two scales)
where stock SM90 keeps them in BF16. Expect ~2-3% relative error per GEMM, and
gate the `max` profile on bench/quality_gate.py before quoting its numbers.

Selected only when the user asks for it:
    --kernel-config '{"linear_backend_per_quant":{"mxfp8":"deep_gemm"}}'
(or H200OPT_MXFP8_DEEPGEMM=1). Knobs:
  H200OPT_MXFP8_EXCLUDE        regex on the layer prefix that keeps BF16
                               (default "indexer|compressor": they drive top-k selection)
  H200OPT_MXFP8_MAX_REL_ERR    whole-matrix re-block error that forces BF16 (5e-3)
  H200OPT_MXFP8_MAX_GROUP_ERR  worst single 1x32 group error that forces BF16 (0.1)
  H200OPT_MXFP8_STRICT         1 (default): fail startup instead of silently running BF16
                               when the backend is requested but cannot be used
"""

from __future__ import annotations

import os
import re

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used, is_deep_gemm_supported

# Importing this module registers torch.ops.vllm.fp8_gemm_nt_op.
from vllm.model_executor.kernels.linear.scaled_mm import deep_gemm as _dg_kernel_mod  # noqa: F401

from .reblock import BLOCK, MX_GROUP, reblock_mxfp8_to_block128

logger = init_logger(__name__)

_MAX_REL_ERR = float(os.environ.get("H200OPT_MXFP8_MAX_REL_ERR", "5e-3"))
_MAX_GROUP_ERR = float(os.environ.get("H200OPT_MXFP8_MAX_GROUP_ERR", "0.1"))
_EXCLUDE = os.environ.get("H200OPT_MXFP8_EXCLUDE", "indexer|compressor")
_EXCLUDE_RE = re.compile(_EXCLUDE) if _EXCLUDE else None
_FORCE = os.environ.get("H200OPT_MXFP8_DEEPGEMM", "0") == "1"
_STRICT = os.environ.get("H200OPT_MXFP8_STRICT", "1") == "1"

# Load-time summary across layers (logged once per process from the last layer).
_SUMMARY = {"layers": 0, "fp8": 0, "bf16_fallback": 0, "excluded": 0,
            "worst_rel_err": 0.0, "worst_group_err": 0.0, "max_shift": 0}


def conversion_summary() -> dict:
    """Load-time counters, e.g. for tests or a startup assertion."""
    return dict(_SUMMARY)


def _requested() -> bool:
    """Whether the user selected the deep_gemm backend for MXFP8 linears."""
    if _FORCE:
        return True
    try:
        from vllm.model_executor.kernels.linear import _get_linear_backend

        return _get_linear_backend(quantization="mxfp8") == "deep_gemm"
    except Exception:  # noqa: BLE001 - private helper; be permissive
        return False


class H200ReblockMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 linear on SM90: re-blocked 128x128 FP8 weights + DeepGEMM fp8_gemm_nt."""

    supports_pre_processed_weights = False

    def __init__(self, config: Mxfp8LinearLayerConfig):
        super().__init__(config)
        self.use_e8m0 = is_deep_gemm_e8m0_used()
        self.quant_fp8 = QuantFP8(
            static=False,
            group_shape=GroupShape(1, BLOCK),
            use_ue8m0=self.use_e8m0,
            tma_aligned_scales=envs.VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES,
            column_major_scales=True,
        )
        self.bf16_fallback = False

    # This kernel consumes BF16 activations and quantizes them itself (1x128),
    # so upstream must not hand it pre-quantized MXFP8 activations.
    @classmethod
    def executed_activation_key(
        cls, activation_quant_key: QuantKey | None
    ) -> QuantKey | None:
        return None

    @classmethod
    def _supports_quant_scheme(
        cls, weight_key: QuantKey | None, activation_key: QuantKey | None
    ) -> bool:
        return weight_key == kMxfp8Static and activation_key in (None, kMxfp8Dynamic)

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not _requested():
            return False, "not requested (set linear_backend_per_quant.mxfp8=deep_gemm)"
        reason = None
        if not current_platform.is_cuda():
            reason = "CUDA only"
        elif not current_platform.is_device_capability(90):
            reason = "H200 re-blocked MXFP8 kernel targets SM90 only"
        elif not is_deep_gemm_supported():
            reason = "DeepGEMM is not available (deep_gemm import / JIT toolchain)"
        if reason is None:
            return True, None
        if _STRICT:
            # Without this the filtered kernel list falls through to BF16 emulation and
            # a "max" run would silently measure stock BF16 (and lose the FP8 GEMV).
            raise RuntimeError(
                f"h200opt: mxfp8=deep_gemm was requested but cannot run: {reason}. "
                "Fix the environment, drop the kernel-config override, or set "
                "H200OPT_MXFP8_STRICT=0 to accept BF16."
            )
        return False, reason

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        if c.bmm_batch_size is not None:
            return False, "grouped BMM layers keep their own path"
        n, k = c.weight_shape
        # N%128 too: the re-block builds whole 128-row tiles (DeepGEMM itself needs N%64).
        if k % BLOCK or n % BLOCK:
            return False, f"needs N%128==0 and K%128==0, got N={n} K={k}"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        assert weight.dtype == torch.float8_e4m3fn and weight.dim() == 2, weight.dtype
        n, k = weight.shape
        scale = layer.weight_scale.data[:n, : k // MX_GROUP].contiguous()
        prefix = getattr(layer, "prefix", "") or f"[{n}x{k}]"
        _SUMMARY["layers"] += 1

        if _EXCLUDE_RE is not None and _EXCLUDE_RE.search(prefix):
            self.bf16_fallback = True
            _SUMMARY["excluded"] += 1
            replace_parameter(layer, "weight", dequant_mxfp8_to_bf16(weight.contiguous(), scale))
            replace_parameter(layer, "weight_scale", scale)
            self._log_progress()
            return

        q, s, stats = reblock_mxfp8_to_block128(weight, scale)
        assert stats is not None
        _SUMMARY["worst_rel_err"] = max(_SUMMARY["worst_rel_err"], stats.rel_err)
        _SUMMARY["worst_group_err"] = max(_SUMMARY["worst_group_err"], stats.max_group_rel_err)
        _SUMMARY["max_shift"] = max(_SUMMARY["max_shift"], stats.max_shift)

        if stats.rel_err > _MAX_REL_ERR or stats.max_group_rel_err > _MAX_GROUP_ERR:
            # Keep accuracy: this layer's 32x32 scales spread too far apart to share one
            # 128x128 scale. Fall back to the stock BF16 behaviour for it.
            self.bf16_fallback = True
            _SUMMARY["bf16_fallback"] += 1
            logger.warning(
                "h200opt: MXFP8 layer %s re-block rel_err=%.2e (limit %.1e), worst 1x32 "
                "group err=%.2e (limit %.2f), max shift %d; keeping BF16 weights for it.",
                prefix, stats.rel_err, _MAX_REL_ERR, stats.max_group_rel_err,
                _MAX_GROUP_ERR, stats.max_shift,
            )
            replace_parameter(layer, "weight", dequant_mxfp8_to_bf16(weight.contiguous(), scale))
            replace_parameter(layer, "weight_scale", scale)
        else:
            _SUMMARY["fp8"] += 1
            dg_w, dg_s = deepgemm_post_process_fp8_weight_block(
                wq=q,
                ws=s,
                quant_block_shape=(BLOCK, BLOCK),
                use_e8m0=self.use_e8m0,
            )
            replace_parameter(layer, "weight", dg_w)
            replace_parameter(layer, "weight_scale", dg_s)
            layer.weight_block_size = [BLOCK, BLOCK]
            # Lets vLLM's DeepGEMM warmup JIT-compile this shape at startup
            # (see vllm_plugin._patch_deep_gemm_warmup).
            layer.deep_gemm_warmup_provider = self

        self._log_progress()

    @staticmethod
    def _log_progress() -> None:
        if _SUMMARY["layers"] % 64 == 1:
            logger.info(
                "h200opt: MXFP8 dense layers so far: %d (%d FP8, %d BF16 by error, %d BF16 "
                "excluded by H200OPT_MXFP8_EXCLUDE=%r); worst rel_err=%.2e, worst group "
                "err=%.2e, max shift=%d",
                _SUMMARY["layers"], _SUMMARY["fp8"], _SUMMARY["bf16_fallback"],
                _SUMMARY["excluded"], _EXCLUDE, _SUMMARY["worst_rel_err"],
                _SUMMARY["worst_group_err"], _SUMMARY["max_shift"],
            )

    def get_deep_gemm_warmup_weights(
        self, layer: torch.nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return layer.weight, layer.weight_scale

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        if self.bf16_fallback:
            return torch.nn.functional.linear(x, weight.to(x.dtype), bias)

        n = weight.shape[0]
        x2 = x.reshape(-1, x.shape[-1])
        out_shape = (*x.shape[:-1], n)
        if x2.shape[0] == 0:
            return x2.new_empty(out_shape, dtype=torch.bfloat16)
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        q_input, input_scale = self.quant_fp8(x2)
        output = torch.empty((x2.shape[0], n), dtype=torch.bfloat16, device=x2.device)
        torch.ops.vllm.fp8_gemm_nt_op(
            q_input, input_scale, weight, layer.weight_scale, output, self.use_e8m0
        )
        if bias is not None:
            output = output + bias
        return output.view(out_shape).to(x.dtype)
