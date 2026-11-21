#!/usr/bin/env bash
# vLLM launch profiles for DeepSeek-V4.1-Flash-1017 on 4x H200.
#
#   serve/vllm/launch.sh <profile>           # runs `vllm serve` in the foreground
#   DRY_RUN=1 serve/vllm/launch.sh <profile> # print the command only
#
# Profiles form a ladder. Each one adds one lever, so an A/B run tells you
# what each lever is worth:
#
#   baseline        vLLM recipe default for H200: TP4 + Engram on host. No spec decode.
#   tp4-spec        + DSpark (5 drafts, adaptive verification) + 128K context + bigger batch
#   dep4            DP4 attention + EP4 experts (default SM90 MoE: Humming W4A8)
#   dep4-mega       + DeepGEMM SM90 MegaMoE (fused EP dispatch/GEMM/combine)
#   dep4-mega-spec  + DSpark
#   max             + native FP8 dense GEMMs (h200opt plugin)       <- expected best
#   max-nospec      max without DSpark (for very high concurrency, where verify compute dominates)
#
# Common env:
#   MODEL_PATH       local checkpoint dir (default: deepseek-ai/DeepSeek-V4.1-Flash-1017)
#   PORT             8000
#   MAX_MODEL_LEN    131072 (baseline keeps the model default unless set explicitly)
#   MAX_NUM_SEQS     per engine; default 256 (DP: per DP rank)
#   MAX_BATCHED      max-num-batched-tokens per engine, default 8192
#   GPU_UTIL         0.92
#   SPEC_K           DSpark drafts, default 5
#   SPEC_SCHEDULE    JSON list for num_speculative_tokens_per_batch_size, e.g. '[[1,64,5],[65,160,3],[161,1024,1]]'
#   ADAPTIVE         1 (adaptive verification on) / 0
#   TEXT_ONLY        1 -> --language-model-only (drops the vision tower)
#   INDEXER_BF16     1 -> BF16 indexer logits (halves the [tokens x max_len] logits buffer)
#   EXTRA_ARGS       appended verbatim
set -euo pipefail

PROFILE=${1:?usage: launch.sh <profile>}
MODEL_PATH=${MODEL_PATH:-deepseek-ai/DeepSeek-V4.1-Flash-1017}
SERVED_NAME=${SERVED_NAME:-deepseek-ai/DeepSeek-V4.1-Flash}
PORT=${PORT:-8000}
GPU_UTIL=${GPU_UTIL:-0.92}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
MAX_BATCHED=${MAX_BATCHED:-8192}
SPEC_K=${SPEC_K:-5}
ADAPTIVE=${ADAPTIVE:-1}
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

# Batch / graph sizing shared by the tuned profiles.
# CUDA graphs must cover the largest decode step: seqs * (1 + drafts).
tuned_sizing() {
  local tokens_per_seq=$1
  local cap=$(( MAX_NUM_SEQS * tokens_per_seq ))
  (( cap > 8192 )) && cap=8192
  echo --max-model-len "${MAX_MODEL_LEN:-131072}" \
       --max-num-seqs "$MAX_NUM_SEQS" \
       --max-num-batched-tokens "$MAX_BATCHED" \
       --max-cudagraph-capture-size "$cap" \
       --gpu-memory-utilization "$GPU_UTIL"
}

KERNEL_CFG=""
ATTN_CFG=""
[[ "${INDEXER_BF16:-0}" == 1 ]] && ATTN_CFG='{"indexer_logits_dtype":"bfloat16"}'
MEGA='{"moe_backend":"deep_gemm_mega_moe"}'
FP8LIN='{"linear_backend_per_quant":{"mxfp8":"deep_gemm"}}'

case "$PROFILE" in
  baseline)
    # Verbatim vLLM recipe for H200 (single_node_tp, TP4, Engram CPU offload).
    ARGS=(--tensor-parallel-size 4 --engram-config '{"cpu_offload":true}')
    [[ -n "${MAX_MODEL_LEN:-}" ]] && ARGS+=(--max-model-len "$MAX_MODEL_LEN")
    ;;
  tp4-spec)
    ARGS=(--tensor-parallel-size 4 --engram-config '{"cpu_offload":true,"use_thp":true}'
          $(tuned_sizing $((SPEC_K + 1))) --speculative-config "$(spec_cfg)")
    ;;
  dep4)
    ARGS=(--tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel
          --engram-config '{"cpu_offload":true}'
          $(tuned_sizing 1))
    ;;
  dep4-mega)
    ARGS=(--tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel
          --engram-config '{"cpu_offload":true}'
          $(tuned_sizing 1))
    KERNEL_CFG=$MEGA
    ;;
  dep4-mega-spec)
    ARGS=(--tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel
          --engram-config '{"cpu_offload":true}'
          $(tuned_sizing $((SPEC_K + 1))) --speculative-config "$(spec_cfg)")
    KERNEL_CFG=$MEGA
    ;;
  max)
    ARGS=(--tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel
          --engram-config '{"cpu_offload":true}'
          $(tuned_sizing $((SPEC_K + 1))) --speculative-config "$(spec_cfg)")
    KERNEL_CFG=$(json_merge "$MEGA" "$FP8LIN")
    ;;
  max-nospec)
    ARGS=(--tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel
          --engram-config '{"cpu_offload":true}'
          $(tuned_sizing 1))
    KERNEL_CFG=$(json_merge "$MEGA" "$FP8LIN")
    ;;
  *)
    echo "unknown profile: $PROFILE" >&2; exit 2 ;;
esac

[[ -n "$KERNEL_CFG" ]] && ARGS+=(--kernel-config "$KERNEL_CFG")
[[ -n "$ATTN_CFG" ]] && ARGS+=(--attention-config "$ATTN_CFG")
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && ARGS+=($EXTRA_ARGS)

CMD=(vllm serve "${COMMON[@]}" "${ARGS[@]}")
mkdir -p "$ROOT/results"
{
  printf 'profile=%s\n' "$PROFILE"
  env | grep -E '^(VLLM_|H200OPT_|CUDA_VISIBLE|PYTORCH_CUDA|DG_)' | sort
  printf '%q ' "${CMD[@]}"; echo
} > "$ROOT/results/.last_server_cmd"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
exec "${CMD[@]}"
