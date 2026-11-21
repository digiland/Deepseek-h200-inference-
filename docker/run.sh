#!/usr/bin/env bash
# Build and run a serving container.
#
#   docker/run.sh build  vllm|sglang                 # build h200/<engine>:latest
#   docker/run.sh up     vllm|sglang <profile>       # start detached, port 8000
#   docker/run.sh logs   vllm|sglang                 # follow logs
#   docker/run.sh down   vllm|sglang
#
# Env:
#   MODEL_DIR     host path of the downloaded checkpoint (required for `up`)
#   GPUS          "0,1,2,3" (default)
#   CACHE_DIR     host dir for JIT/compile caches (default ~/.cache/dsv41-h200), keep it:
#                 DeepGEMM/FlashInfer/Triton compiles take many minutes the first time
#   VLLM_IMAGE / SGLANG_IMAGE   base image override at build time
#   PASS_ENV      space-separated extra env var names to forward (e.g. "H200OPT_SM90_MHC_MAX_TOKENS SPEC_K")
set -euo pipefail
ACTION=${1:?build|up|logs|down}; ENGINE=${2:?vllm|sglang}; PROFILE=${3:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NAME=dsv41-$ENGINE
IMAGE=h200/$ENGINE:latest
GPUS=${GPUS:-0,1,2,3}
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/dsv41-h200}

case "$ACTION" in
  build)
    args=()
    [[ -n "${VLLM_IMAGE:-}" ]] && args+=(--build-arg "VLLM_IMAGE=$VLLM_IMAGE")
    [[ -n "${SGLANG_IMAGE:-}" ]] && args+=(--build-arg "SGLANG_IMAGE=$SGLANG_IMAGE")
    docker build "${args[@]}" -f "$ROOT/docker/Dockerfile.$ENGINE" -t "$IMAGE" "$ROOT"
    ;;
  up)
    : "${MODEL_DIR:?set MODEL_DIR to the local checkpoint directory}"
    : "${PROFILE:?profile required, e.g. max / baseline / dpa4}"
    mkdir -p "$CACHE_DIR" "$ROOT/results"
    envs=(-e MODEL_PATH=/model -e CUDA_VISIBLE_DEVICES="$GPUS" -e HF_HUB_OFFLINE=1)
    for v in PORT MAX_MODEL_LEN MAX_NUM_SEQS MAX_BATCHED GPU_UTIL SPEC_K SPEC_SCHEDULE ADAPTIVE \
             TEXT_ONLY INDEXER_BF16 EXTRA_ARGS CONTEXT_LEN MAX_RUNNING MEM_FRAC CHUNK GRAPH_BS \
             DSPARK_BLOCK MOE_PRECISION ${PASS_ENV:-}; do
      [[ -n "${!v:-}" ]] && envs+=(-e "$v=${!v}")
    done
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    # --privileged + --ipc=host + --pid=host: CUDA IPC / NVLink multicast (MegaMoE symmetric
    # memory, fused all-reduce) and the 189 GiB pinned Engram host tables (memlock unlimited).
    # --privileged ignores --gpus device=..., so GPUs are selected via CUDA_VISIBLE_DEVICES.
    docker run -d --name "$NAME" --gpus all --privileged --ipc=host --pid=host --network host \
      --ulimit memlock=-1 --ulimit stack=67108864 --shm-size 64g \
      -v "$MODEL_DIR":/model:ro \
      -v "$CACHE_DIR":/root/.cache \
      -v "$ROOT/results":/opt/h200/results \
      "${envs[@]}" "$IMAGE" "$PROFILE"
    echo "started $NAME ($PROFILE); follow with: docker/run.sh logs $ENGINE"
    ;;
  logs) docker logs -f "$NAME" ;;
  down) docker rm -f "$NAME" ;;
  *) echo "unknown action $ACTION" >&2; exit 2 ;;
esac
