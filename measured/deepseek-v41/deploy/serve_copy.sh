#!/bin/bash
# Start ONE copy of DeepSeek-V4.1-Flash on 4 GPUs.  usage: serve_copy.sh <best|tput> <port> <gpus>
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd); . "$D/node.env"
PROFILE=$1 PORT=$2 GPUS=$3
VP=/usr/local/lib/python3.12/dist-packages/vllm
ENV=(-e VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3","quant_mode":"dynamic_token"}')
MNT=(); ARGS=()
if [ "$PROFILE" = best ]; then
  ENV+=(-e V41_HUMMING_W13_OVERRIDE='{"num_ctas_per_sm":2}' -e V41_HUMMING_W2_OVERRIDE='{"num_ctas_per_sm":2}'
        -e V41_HUMMING_OVERRIDE_MAX_M=2208 -e V41_PREFILL_INTERVAL=8)
  MNT=(-m "$D/patches/fused_humming_moe.py:$VP/model_executor/layers/fused_moe/experts/fused_humming_moe.py"
       -m "$D/patches/core.py:$VP/v1/engine/core.py")
  ARGS=(--max-num-seqs 256 --max-cudagraph-capture-size 2048
        --speculative-config '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"block"}')
elif [ "$PROFILE" = tput ]; then
  ARGS=(--max-num-seqs 1024 --max-cudagraph-capture-size 1024)
else echo "profile must be best or tput"; exit 1; fi
mkdir -p "$V41_LOGS" "$V41_CACHE"/{home,vllm,triton,xdg}
exec enroot start -m /data:/data -m "$D:$D" -m "$V41_CACHE/home:/home/ubuntu" \
  -e HOME=/home/ubuntu -e VLLM_CACHE_ROOT="$V41_CACHE/vllm" -e TRITON_CACHE_DIR="$V41_CACHE/triton" \
  -e XDG_CACHE_HOME="$V41_CACHE/xdg" -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_USE_RUST_FRONTEND=1 \
  -e CUDA_VISIBLE_DEVICES="$GPUS" "${ENV[@]}" "${MNT[@]}" "$V41_CONTAINER" \
  python3 -m vllm.entrypoints.openai.api_server --model "$V41_MODEL" \
    --served-model-name deepseek-ai/DeepSeek-V4.1-Flash-1017 --host "$V41_HOST" --port "$PORT" \
    --tensor-parallel-size 4 --engram-config '{"cpu_offload":true}' \
    --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41 \
    --mm-encoder-tp-mode data --max-model-len 131072 --kernel-config '{"enable_flashinfer_autotune":true}' \
    "${ARGS[@]}" > "$V41_LOGS/serve-$PORT.log" 2>&1
