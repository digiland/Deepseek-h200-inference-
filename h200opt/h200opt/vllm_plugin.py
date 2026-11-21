"""vLLM general plugin: H200 (SM90) optimizations for DeepSeek-V4.1-Flash.

vLLM loads this in every process (API server, engine core, workers) through
the `vllm.general_plugins` entry point. A patch whose target moved in a newer
vLLM logs a warning and is skipped. That is NOT the same as "stock vLLM still
runs": if the MXFP8 kernel failed to register and the deep_gemm backend is
requested, V4.1's grouped wo_a layer has no SM90 kernel left and startup fails.
The Docker build runs the registration (`python -m h200opt.check`), so a drifted
base image fails at build time instead.

Patches (each opt-in, so the defaults equal stock vLLM):

1. MXFP8 dense linears -> re-blocked FP8 + DeepGEMM on SM90
   enable: --kernel-config '{"linear_backend_per_quant":{"mxfp8":"deep_gemm"}}'
2. SM90 mHC / GEMV row thresholds, which were tuned on H100 TP8 and can be
   retuned for H200:
   H200OPT_SM90_MHC_MAX_TOKENS, H200OPT_SM90_MHC_ALL_REDUCE_MAX_TOKENS,
   H200OPT_GEMV_MAX_ROWS
3. Fused all-reduce + mHC boundary at TP4 on SM90 (stock vLLM allows TP8 only):
   H200OPT_SM90_MHC_ALLREDUCE_TP=4

Set H200OPT_DISABLE=1 to load nothing.
"""

from __future__ import annotations

import os

_APPLIED = False


def _log():
    from vllm.logger import init_logger

    return init_logger("h200opt")


def _int_env(name: str) -> int | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return None
    return int(v)


def _register_mxfp8_kernel() -> None:
    from vllm.model_executor.kernels import linear as lk
    from vllm.platforms import PlatformEnum

    from .vllm_mxfp8_sm90 import H200ReblockMxfp8LinearKernel

    kernels = lk._POSSIBLE_MXFP8_KERNELS.setdefault(PlatformEnum.CUDA, [])
    if H200ReblockMxfp8LinearKernel not in kernels:
        # Front of the list: is_supported() only says yes on SM90 *and* when the
        # user asked for the deep_gemm MXFP8 backend, so auto selection elsewhere
        # is unchanged.
        kernels.insert(0, H200ReblockMxfp8LinearKernel)
    dg = lk._LINEAR_BACKEND_KERNEL_MAP.setdefault("deep_gemm", set())
    dg.add(H200ReblockMxfp8LinearKernel)
    # Shapes the kernel cannot take (N%64, K%128) fall back to the BF16 emulation
    # kernel instead of failing startup.
    dg.add(lk.EmulationMxfp8LinearKernel)


def _patch_deep_gemm_warmup() -> None:
    """Let vLLM's DeepGEMM JIT warmup see our kernel's layers (it isinstance-checks)."""
    from vllm.model_executor.warmup import deep_gemm_warmup as w

    from .vllm_mxfp8_sm90 import H200ReblockMxfp8LinearKernel

    orig = w._deep_gemm_linear_data
    if getattr(orig, "_h200opt", False):
        return

    def _deep_gemm_linear_data(module):
        provider = getattr(module, "deep_gemm_warmup_provider", None)
        if isinstance(provider, H200ReblockMxfp8LinearKernel):
            if provider.bf16_fallback:
                return None
            return provider.get_deep_gemm_warmup_weights(module)
        return orig(module)

    _deep_gemm_linear_data._h200opt = True  # type: ignore[attr-defined]
    w._deep_gemm_linear_data = _deep_gemm_linear_data


def _patch_thresholds() -> None:
    mhc_max = _int_env("H200OPT_SM90_MHC_MAX_TOKENS")
    ar_max = _int_env("H200OPT_SM90_MHC_ALL_REDUCE_MAX_TOKENS")
    gemv_max = _int_env("H200OPT_GEMV_MAX_ROWS")
    ar_tp = _int_env("H200OPT_SM90_MHC_ALLREDUCE_TP")
    if mhc_max is None and ar_max is None and gemv_max is None and ar_tp is None:
        return
    log = _log()
    if mhc_max is not None or ar_max is not None or ar_tp is not None:
        from vllm.models.deepseek_v41.nvidia.ops import mhc

        if mhc_max is not None:
            mhc.SM90_MHC_MAX_TOKENS = mhc_max
            if hasattr(mhc.ampere_mhc_max_tokens, "cache_clear"):
                mhc.ampere_mhc_max_tokens.cache_clear()
            log.info("h200opt: SM90_MHC_MAX_TOKENS=%d", mhc_max)
        if ar_max is not None:
            mhc.SM90_MHC_ALL_REDUCE_MAX_TOKENS = ar_max
            log.info("h200opt: SM90_MHC_ALL_REDUCE_MAX_TOKENS=%d", ar_max)
        if ar_tp is not None:
            _patch_mhc_all_reduce_tp(mhc, ar_tp)
    if gemv_max is not None:
        from vllm.models.deepseek_v41.nvidia import quantization as q

        q.GEMV_MAX_ROWS = gemv_max
        log.info("h200opt: GEMV_MAX_ROWS=%d", gemv_max)


def _patch_mhc_all_reduce_tp(mhc, tp: int) -> None:
    """Allow the fused all-reduce + mHC boundary at `tp` on SM90.

    Mirrors `supports_mhc_all_reduce` with the SM90 TP size replaced. It still
    requires NVLink multicast (checked through the custom all-reduce's MNNVL
    buffers), which H200 HGX boards with NVSwitch provide.
    """
    from typing import cast

    from vllm.distributed import get_tp_group
    from vllm.platforms import current_platform

    orig = mhc.supports_mhc_all_reduce
    if getattr(orig, "_h200opt", False):
        return

    def supports_mhc_all_reduce(vllm_config):
        if not current_platform.is_device_capability(90):
            return orig(vllm_config)
        parallel = vllm_config.parallel_config
        config = vllm_config.model_config.hf_config
        if (
            parallel.tensor_parallel_size != tp
            or parallel.enable_expert_parallel
            or config.hidden_size != 5120
            or config.hc_mult != 4
        ):
            return orig(vllm_config)
        comm = cast("object", get_tp_group().device_communicator).ca_comm  # type: ignore[attr-defined]
        return comm is not None and bool(comm.mnnvl_lamport_ag_multicast_ptr)

    supports_mhc_all_reduce._h200opt = True  # type: ignore[attr-defined]
    mhc.supports_mhc_all_reduce = supports_mhc_all_reduce
    # Importing ops.mhc runs the deepseek_v41 package __init__, which already imported
    # nvidia/model.py and specialized_decode with `from .ops.mhc import
    # supports_mhc_all_reduce`. Rebind those names too, or the patch is dead.
    import sys

    rebound = []
    for modname in (
        "vllm.models.deepseek_v41.nvidia.model",
        "vllm.models.deepseek_v41.nvidia.specialized_decode",
    ):
        mod = sys.modules.get(modname)
        if mod is None:
            try:
                import importlib

                mod = importlib.import_module(modname)
            except Exception:  # noqa: BLE001
                continue
        if getattr(mod, "supports_mhc_all_reduce", None) is orig:
            mod.supports_mhc_all_reduce = supports_mhc_all_reduce
            rebound.append(modname.rsplit(".", 1)[-1])
    _log().info("h200opt: fused all-reduce + mHC enabled for TP%d on SM90 (rebound in %s)",
                tp, ", ".join(rebound) or "nothing")


def register() -> None:
    """Entry point (`vllm.general_plugins`)."""
    global _APPLIED
    if _APPLIED or os.environ.get("H200OPT_DISABLE") == "1":
        return
    _APPLIED = True
    log = _log()
    for name, fn in (
        ("mxfp8-sm90-deepgemm", _register_mxfp8_kernel),
        ("deep-gemm-warmup", _patch_deep_gemm_warmup),
        ("sm90-thresholds", _patch_thresholds),
    ):
        try:
            fn()
            log.debug("h200opt: applied %s", name)
        except Exception as e:  # noqa: BLE001 - never break the server
            log.warning("h200opt: skipped patch %s (%s: %s)", name, type(e).__name__, e)
