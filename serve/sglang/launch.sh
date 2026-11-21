#!/usr/bin/env bash
# SGLang launch profiles for DeepSeek-V4.1-Flash-1017 on 4x H200.
#
#   serve/sglang/launch.sh <profile>
#   DRY_RUN=1 serve/sglang/launch.sh <profile>
#
#   baseline      InferenceX's H200 TP4 recipe: TP4, Marlin W4A16, DSpark 5, graph bs<=64
#   dpa4          attention DP4 (--attn-dp-size 4; SGLang pads TP-sharded heads to 64, DP avoids it)
#                 + EP4 + FlashInfer MXFP4 MoE with FP8 activations (Humming W4A8) + DSpark,
#                 graphs to bs 256. NOTE: bounded replay with attention DP requires
#                 moe_a2a_backend none|megamoe; with "none" the MoE runs EP over all-gather.
#                 Validate with DRY_RUN inside the image first:
#                   docker run --rm --entrypoint python3 h200/sglang:latest -m sglang.launch_server --help
#   dpa4-nospec   dpa4 without DSpark (the SGLang docs say spec decode stops paying off at high batch)
#   dpa4-hicache  dpa4 + hierarchical (host-memory) prefix cache, as in InferenceX AgentX DEP4
#
# Env: MODEL_PATH, PORT (default 8000), CONTEXT_LEN (131072), MAX_RUNNING (per DP rank, 256),
#      MEM_FRAC (0.85), CHUNK (8192), GRAPH_BS (256), DSPARK_BLOCK (5), MOE_PRECISION (fp8|default),
#      EXTRA_ARGS
set -euo pipefail

PROFILE=${1:?usage: launch.sh <profile>}
MODEL_PATH=${MODEL_PATH:-deepseek-ai/DeepSeek-V4.1-Flash-1017}
SERVED_NAME=${SERVED_NAME:-deepseek-ai/DeepSeek-V4.1-Flash}
PORT=${PORT:-8000}
CONTEXT_LEN=${CONTEXT_LEN:-131072}
MAX_RUNNING=${MAX_RUNNING:-256}
MEM_FRAC=${MEM_FRAC:-0.85}
CHUNK=${CHUNK:-8192}
GRAPH_BS=${GRAPH_BS:-256}
DSPARK_BLOCK=${DSPARK_BLOCK:-5}
MOE_PRECISION=${MOE_PRECISION:-fp8}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONUNBUFFERED=1
export SGLANG_TIMEOUT_KEEP_ALIVE=${SGLANG_TIMEOUT_KEEP_ALIVE:-900}

COMMON=(
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_NAME"
  --port "$PORT" --host 0.0.0.0
  --trust-remote-code
  --attention-backend dsv4
  --reasoning-parser auto --tool-call-parser auto
  --engram-config '{"placement":"cpu"}'
  --watchdog-timeout 3600
  --enable-metrics
)
DSPARK=(--speculative-algorithm DSPARK --speculative-dspark-block-size "$DSPARK_BLOCK")

DPA=(
  # SGLang 1.0: attention DP inside the TP group is --attn-dp-size; --dp-size now means
  # whole-model replicas (4 x TP4 = 16 GPUs) and --enable-dp-attention is a removed flag.
  --tp-size 4 --attn-dp-size 4 --ep-size 4
  --moe-runner-backend flashinfer_mxfp4 --flashinfer-mxfp4-moe-precision "$MOE_PRECISION"
  --enable-dp-lm-head
  --enable-decoder-swa-bounded-replay
  --context-length "$CONTEXT_LEN"
  --mem-fraction-static "$MEM_FRAC"
  --chunked-prefill-size "$CHUNK"
  --max-running-requests "$(( MAX_RUNNING * 4 ))"
  --cuda-graph-max-bs-decode "$GRAPH_BS"
  --swa-prefix-tails 4096
)

case "$PROFILE" in
  baseline)
    # inferencex-e2e/.../dsv41flash/sglang/h200-fp4-mtp/agentic.yaml (TP4), with the
    # Engram host-table env vars replaced by --engram-config (current main rejects the env vars).
    ARGS=(--tp-size 4 --ep-size 1
          --moe-runner-backend marlin
          --mem-fraction-static 0.70 --chunked-prefill-size 4096
          --prefill-decode-interval 16
          --enable-decoder-swa-bounded-replay --swa-prefix-tails 1024
          "${DSPARK[@]}" --cuda-graph-max-bs-decode 64
          --max-running-requests 64)   # recipe rule: admission <= decode graph batch (64)
    ;;
  dpa4)          ARGS=("${DPA[@]}" "${DSPARK[@]}") ;;
  dpa4-nospec)   ARGS=("${DPA[@]}") ;;
  dpa4-hicache)
    ARGS=("${DPA[@]}" "${DSPARK[@]}"
          --enable-hierarchical-cache --hicache-ratio 2 --hicache-io-backend direct
          --hicache-mem-layout page_first_direct --hicache-write-policy write_back)
    ;;
  *) echo "unknown profile: $PROFILE" >&2; exit 2 ;;
esac
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && ARGS+=($EXTRA_ARGS)

CMD=(python3 -m sglang.launch_server "${COMMON[@]}" "${ARGS[@]}")
mkdir -p "$ROOT/results"
{ printf 'profile=sglang-%s\n' "$PROFILE"; printf '%q ' "${CMD[@]}"; echo; } > "$ROOT/results/.last_server_cmd"
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
exec "${CMD[@]}"
