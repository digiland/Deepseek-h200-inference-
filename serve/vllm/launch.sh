#!/usr/bin/env bash
# vLLM launch profiles for DeepSeek-V4.1-Flash-1017 on 4x H200.
#
#   serve/vllm/launch.sh <profile>           # runs `vllm serve` in the foreground
#   DRY_RUN=1 serve/vllm/launch.sh <profile> # print the command only
#
# Baselines. Claim the 2x against the strongest of these, not the verbatim recipe:
#   baseline         vLLM recipe for H200, verbatim: TP4 + Engram on host, model-default
#                    context (1M) and scheduler defaults (128 seqs). No spec decode.
#   baseline-spec    the recipe plus its own `spec_decoding` feature (DSpark 5, adaptive)
#   tp4-tuned        TP4 with the same scheduler budget as the tuned DP profiles
#                    (128K context, NODE_SEQS sequences, graphs sized to match). No spec.
#   tp4-tuned-spec   tp4-tuned + DSpark
# The ladder. Each step adds one lever on top of the previous one:
#   dep4             DP4 attention + EP4 experts (MoE: Humming W4A8; all2all: auto =
#                    allgather_reducescatter on H200; try EXTRA_ARGS="--all2all-backend alltoall_batched")
#   dep4-mega        + DeepGEMM SM90 MegaMoE (fused EP dispatch/GEMM/combine)
#   dep4-mega-spec   + DSpark
#   max              + native FP8 dense GEMMs (h200opt plugin; strict; gate with bench/quality_gate.py)
#   max-nospec       max without DSpark (very high concurrency; raise NODE_SEQS to 2048-4096)
#
# Env:
#   MODEL_PATH       checkpoint dir (default: deepseek-ai/DeepSeek-V4.1-Flash-1017)
#   PORT             8000
#   NODE_SEQS        max concurrent sequences for the whole node (default 1024). DP profiles
#                    give each of the 4 ranks NODE_SEQS/4, TP4 profiles get all of it, so
#                    TP4 and DP4 rungs have the same batch budget.
#   MAX_MODEL_LEN    131072 for tuned profiles (baseline/baseline-spec keep the model default
#                    unless set)
#   MAX_BATCHED      max-num-batched-tokens per engine (default 8192)
#   GPU_UTIL         0.92
#   SPEC_K           DSpark drafts (default 5)
#   SPEC_SCHEDULE    num_speculative_tokens_per_batch_size, e.g. '[[1,64,5],[65,192,3],[193,1024,1]]'
#   ADAPTIVE         1 (adaptive verification) / 0
#   ENGRAM_CFG       engram config JSON (default {"cpu_offload":true})
#   TEXT_ONLY        1 -> --language-model-only (drops the vision tower)
#   INDEXER_BF16     1 -> BF16 indexer logits
#   EPLB             1 -> --enable-eplb (DP profiles)
#   EXTRA_ARGS       appended verbatim, e.g. --attention-config '{"backend":"FLASHMLA_SPARSE_DSV41"}'
set -euo pipefail

PROFILE=${1:?usage: launch.sh <profile>}
MODEL_PATH=${MODEL_PATH:-deepseek-ai/DeepSeek-V4.1-Flash-1017}
SERVED_NAME=${SERVED_NAME:-deepseek-ai/DeepSeek-V4.1-Flash}
PORT=${PORT:-8000}
GPU_UTIL=${GPU_UTIL:-0.92}
NODE_SEQS=${NODE_SEQS:-1024}
MAX_BATCHED=${MAX_BATCHED:-8192}
SPEC_K=${SPEC_K:-5}
ADAPTIVE=${ADAPTIVE:-1}
ENGRAM_CFG=${ENGRAM_CFG:-'{"cpu_offload":true}'}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-7200}
export VLLM_USE_RUST_FRONTEND=${VLLM_USE_RUST_FRONTEND:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
# Persist DeepGEMM / FlashInfer / Triton JIT caches between restarts (compiles take minutes).
export DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-$HOME/.cache/deep_gemm}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}

json_merge() {  # merge JSON objects given as args -> one object
  python3 -c 'import json,sys; d={}; [d.update(json.loads(a)) for a in sys.argv[1:] if a]; print(json.dumps(d, separators=(",",":")))' "$@"
}

COMMON=(
  "$MODEL_PATH"
  --served-model-name "$SERVED_NAME"
  --port "$PORT"
  --tokenizer-mode deepseek_v41
  --reasoning-parser deepseek_v41
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice
  --safetensors-load-strategy prefetch
  --enable-prefix-caching
  --disable-uvicorn-access-log
)
if [[ "${TEXT_ONLY:-0}" == 1 ]]; then
  COMMON+=(--language-model-only)
else
  # Keep multimodal; run the small ViT data-parallel instead of TP-sharded.
  COMMON+=(--mm-encoder-tp-mode data)
fi

spec_cfg() {
  local adaptive=true; [[ "$ADAPTIVE" == 0 ]] && adaptive=false
  local cfg="{\"method\":\"dspark\",\"num_speculative_tokens\":$SPEC_K,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":$adaptive}"
  if [[ -n "${SPEC_SCHEDULE:-}" ]]; then
    cfg=$(json_merge "$cfg" "{\"num_speculative_tokens_per_batch_size\":$SPEC_SCHEDULE}")
  fi
  echo "$cfg"
}

# Scheduler budget + CUDA graph coverage for one engine with `seqs` sequences.
# Graphs must cover the largest decode step: seqs * (1 + drafts).
sizing() {  # seqs tokens_per_seq
  local cap=$(( $1 * $2 ))
  (( cap > 8192 )) && cap=8192
  echo --max-model-len "${MAX_MODEL_LEN:-131072}" \
       --max-num-seqs "$1" \
       --max-num-batched-tokens "$MAX_BATCHED" \
       --max-cudagraph-capture-size "$cap" \
       --gpu-memory-utilization "$GPU_UTIL"
}
TP4_SEQS=$NODE_SEQS
DP_SEQS=$(( NODE_SEQS / 4 ))
TP4=(--tensor-parallel-size 4 --engram-config "$ENGRAM_CFG")
DEP4=(--tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel --engram-config "$ENGRAM_CFG")
[[ "${EPLB:-0}" == 1 ]] && DEP4+=(--enable-eplb)

KERNEL_CFG=""
ATTN_CFG=""
[[ "${INDEXER_BF16:-0}" == 1 ]] && ATTN_CFG='{"indexer_logits_dtype":"bfloat16"}'
MEGA='{"moe_backend":"deep_gemm_mega_moe"}'
FP8LIN='{"linear_backend_per_quant":{"mxfp8":"deep_gemm"}}'

case "$PROFILE" in
  baseline)
    ARGS=(--tensor-parallel-size 4 --engram-config '{"cpu_offload":true}')
    [[ -n "${MAX_MODEL_LEN:-}" ]] && ARGS+=(--max-model-len "$MAX_MODEL_LEN")
    ;;
  baseline-spec)
    # recipe yaml `features.spec_decoding` (NVIDIA row), verbatim
    ARGS=(--tensor-parallel-size 4 --engram-config '{"cpu_offload":true}'
          --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}')
    [[ -n "${MAX_MODEL_LEN:-}" ]] && ARGS+=(--max-model-len "$MAX_MODEL_LEN")
    ;;
  tp4-tuned)       ARGS=("${TP4[@]}" $(sizing "$TP4_SEQS" 1)) ;;
  tp4-tuned-spec)  ARGS=("${TP4[@]}" $(sizing "$TP4_SEQS" $((SPEC_K + 1))) --speculative-config "$(spec_cfg)") ;;
  dep4)            ARGS=("${DEP4[@]}" $(sizing "$DP_SEQS" 1)) ;;
  dep4-mega)       ARGS=("${DEP4[@]}" $(sizing "$DP_SEQS" 1)); KERNEL_CFG=$MEGA ;;
  dep4-mega-spec)  ARGS=("${DEP4[@]}" $(sizing "$DP_SEQS" $((SPEC_K + 1))) --speculative-config "$(spec_cfg)")
                   KERNEL_CFG=$MEGA ;;
  max)             ARGS=("${DEP4[@]}" $(sizing "$DP_SEQS" $((SPEC_K + 1))) --speculative-config "$(spec_cfg)")
                   KERNEL_CFG=$(json_merge "$MEGA" "$FP8LIN") ;;
  max-nospec)      ARGS=("${DEP4[@]}" $(sizing "$DP_SEQS" 1))
                   KERNEL_CFG=$(json_merge "$MEGA" "$FP8LIN") ;;
  *) echo "unknown profile: $PROFILE" >&2; exit 2 ;;
esac

[[ -n "$KERNEL_CFG" ]] && ARGS+=(--kernel-config "$KERNEL_CFG")
[[ -n "$ATTN_CFG" ]] && ARGS+=(--attention-config "$ATTN_CFG")
# EXTRA_ARGS may contain quoted JSON: let bash parse it.
if [[ -n "${EXTRA_ARGS:-}" ]]; then eval "EXTRA=($EXTRA_ARGS)"; ARGS+=("${EXTRA[@]}"); fi

CMD=(vllm serve "${COMMON[@]}" "${ARGS[@]}")
mkdir -p "$ROOT/results"
{
  printf 'profile=%s\n' "$PROFILE"
  python3 -c 'import vllm,sys; print("vllm", vllm.__version__)' 2>/dev/null || true
  env | grep -E '^(VLLM_|H200OPT_|CUDA_VISIBLE|PYTORCH_CUDA|DG_|NODE_SEQS|SPEC_|EPLB)' | sort
  printf '%q ' "${CMD[@]}"; echo
} > "$ROOT/results/.last_server_cmd"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
exec "${CMD[@]}"
