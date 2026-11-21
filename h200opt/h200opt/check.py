"""Build-time / pre-flight check that every h200opt patch still applies to the
installed vLLM:  python -m h200opt.check   (exit 1 on any failure)."""

import sys


def main() -> int:
    from . import vllm_plugin as p

    failures = []
    for name, fn in (("mxfp8 kernel registration", p._register_mxfp8_kernel),
                     ("deep_gemm warmup hook", p._patch_deep_gemm_warmup)):
        try:
            fn()
            print(f"ok   {name}")
        except Exception as e:  # noqa: BLE001
            failures.append(name)
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    try:
        from vllm.models.deepseek_v41.nvidia.ops import mhc  # noqa: F401
        from vllm.models.deepseek_v41.nvidia import quantization as q

        for attr in ("SM90_MHC_MAX_TOKENS", "SM90_MHC_ALL_REDUCE_MAX_TOKENS", "supports_mhc_all_reduce"):
            assert hasattr(mhc, attr), attr
        assert hasattr(q, "GEMV_MAX_ROWS") and hasattr(q, "mxfp8_linear_as_bf16")
        print("ok   SM90 threshold targets present")
    except Exception as e:  # noqa: BLE001
        failures.append("threshold targets")
        print(f"FAIL threshold targets: {type(e).__name__}: {e}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
