#!/bin/bash
# Stage 2 of the DFly2 fine-tune: run V4.1 (TP8, W4A8 like serving) over the training conversations
# and write the hidden states DFly2 reads (layers 19, 27, 31, 35 + last) to disk, no training.
# usage: run_capture.sh <train.jsonl> <run name> [extra key=value overrides]
set -euo pipefail
DATA=$1; NAME=$2; shift 2
F=/data/nvme3/dfly2/ft; R=$F/runs/$NAME; mkdir -p $R
. /data/nvme3/shona-cpt/enroot/env.sh
exec enroot start -m /data/nvme3:/data/nvme3 \
  -m /data/nvme3/shona-cpt/cache/home:/home/ubuntu \
  -e HOME=/home/ubuntu -e HF_HOME=/data/nvme3/shona-cpt/hf_cache \
  -e VLLM_CACHE_ROOT=/data/nvme3/shona-cpt/cache/vllm -e TRITON_CACHE_DIR=/data/nvme3/shona-cpt/cache/triton \
  -e PYTHONPATH=/data/nvme3/dfly2/AngelSpec:$F/pydeps:$F/stubs \
  -e MC_STORE_MEMCPY=0 -e RAY_TMPDIR=$R/ray \
  -e VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3","quant_mode":"dynamic_token"}' \
  -e DSV41_TARGET_MODEL=/data/nvme3/lab/teachers/deepseek-v4.1-flash-1017 \
  -e DSV41_TRAIN_DATA=$DATA -e ANGELSPEC_RUN_DIR=$R \
  vllm bash -c "mkdir -p $R/ray && cd /data/nvme3/dfly2/AngelSpec && python3 -m angelspec.train_entry \
    --config configs/vllm_dsv41_flash_dfly2.yaml \
    online_dump.enabled=true online_dump.dump_only=true \
    online_dump.dump_path=$R/hidden online_dump.frozen_target_path=$R/frozen \
    inference.last_hidden_states_prenorm=true inference.vllm.enable_multimodal=false \
    training.max_seq_length=12288 inference.vllm.extra_args.max_model_len=12288 \
    $*"
