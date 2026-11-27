#!/bin/bash
# DeepSeek-V4.1-Flash on $GPUS with TP $TP (default GPUs 4-7, TP4), same flags as run_final_v41.sh F1 (W4A8 + Humming 2
# CTAs/SM + prompts every 8th step), plus an optional speculative config.
#   MODE=dfly2 NSPEC=5 REJ=standard  -> AngelSpec DFly2 drafter (patched vLLM files mounted)
#   MODE=dspark NSPEC=3              -> V4.1's built-in DSpark (probabilistic, block), as F1
#   EXTRA_ENV="A=1 B=2"              -> extra container env
# Log: /data/nvme3/dfly2/logs/serve-$PORT.log
set -euo pipefail
S=/data/nvme3/shona-cpt; T=/data/nvme3/lab/teachers; W=/data/nvme3/dfly2
M=$T/deepseek-v4.1-flash-1017
PORT=${PORT:-8170}
MODE=${MODE:-dfly2}
NSPEC=${NSPEC:-5}
REJ=${REJ:-standard}
DRAFT_SAMPLE=${DRAFT_SAMPLE:-probabilistic}
. "$S/enroot/env.sh"
VP=/usr/local/lib/python3.12/dist-packages/vllm
CDM=$VP/model_executor/layers/fused_moe/experts
ENVS=(VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3","quant_mode":"dynamic_token"}'
      V41_HUMMING_W13_OVERRIDE='{"num_ctas_per_sm":2}' V41_HUMMING_W2_OVERRIDE='{"num_ctas_per_sm":2}'
      V41_HUMMING_OVERRIDE_MAX_M=2208 V41_PREFILL_INTERVAL=8)
MNTS=("$T/patches/humming_tune/fused_humming_moe.py:$CDM/fused_humming_moe.py"
      "$T/patches/prefill_interval/core.py:$VP/v1/engine/core.py")
if [ "$MODE" = dfly2 ]; then
  SPEC="{\"method\":\"dspark\",\"model\":\"${DFLY2_MODEL:-$W/served/DS-V4.1-Flash-DFly2-B6}\",\"num_speculative_tokens\":$NSPEC,\"draft_sample_method\":\"$DRAFT_SAMPLE\",\"rejection_sample_method\":\"$REJ\"}"
  ENVS+=(PYTHONPATH=$W/pkg DFLY2_TOPK_LAYERS=28,32,36 DFLY2_NORMED_HIDDEN=1)
  MNTS+=("$W/patches/attention.py:$VP/models/deepseek_v41/attention.py"
         "$W/patches/spec_decode__init__.py:$VP/v1/worker/gpu/spec_decode/__init__.py"
         "$W/patches/spec_decode_utils.py:$VP/v1/worker/gpu/spec_decode/utils.py")
elif [ "$MODE" = dspark ]; then
  SPEC="{\"method\":\"dspark\",\"num_speculative_tokens\":$NSPEC,\"draft_sample_method\":\"$DRAFT_SAMPLE\",\"rejection_sample_method\":\"${REJ}\"}"
else
  SPEC=""
fi
for kv in ${EXTRA_ENV:-}; do ENVS+=("$kv"); done
EXTRA=()
for kv in "${ENVS[@]}"; do EXTRA+=(-e "$kv"); done
for m in "${MNTS[@]}"; do EXTRA+=(-m "$m"); done
SPECARGS=()
[ -n "$SPEC" ] && SPECARGS=(--speculative-config "$SPEC")
mkdir -p $W/logs
echo "MODE=$MODE NSPEC=$NSPEC REJ=$REJ SPEC=$SPEC" > $W/logs/serve-$PORT.log
exec enroot start \
  -m /data/nvme3:/data/nvme3 \
  -m "$S/cache/home:/home/ubuntu" \
  -e HOME=/home/ubuntu -e HF_HOME=$S/hf_cache -e VLLM_CACHE_ROOT=$S/cache/vllm \
  -e TRITON_CACHE_DIR=$S/cache/triton -e XDG_CACHE_HOME=$S/cache/xdg \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_USE_RUST_FRONTEND=1 \
  -e CUDA_VISIBLE_DEVICES=${GPUS:-4,5,6,7} \
  "${EXTRA[@]}" \
  vllm \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$M" \
    --served-model-name deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size ${TP:-4} \
    --engram-config '{"cpu_offload":true}' \
    --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v41 --mm-encoder-tp-mode data \
    --max-model-len 131072 --max-num-seqs 256 --max-cudagraph-capture-size 2048 \
    --kernel-config '{"enable_flashinfer_autotune":true}' \
    "${SPECARGS[@]}" "$@" \
    >> $W/logs/serve-$PORT.log 2>&1
