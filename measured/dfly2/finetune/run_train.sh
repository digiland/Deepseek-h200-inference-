#!/bin/bash
# DFly2 fine-tune on DeepSeek V4.1 thinking-mode replies, online on one node:
# V4.1 inference TP4 on 4 GPUs (W4A8, like serving) streams hidden states through Mooncake to
# DFly2 training (FSDP) on the other 4 GPUs. Starts from the released DFly2 weights.
# usage: run_train.sh <train.jsonl> <run name> [extra key=value overrides]
set -euo pipefail
DATA=$1; NAME=$2; shift 2
F=/data/nvme3/dfly2/ft; R=$F/runs/$NAME; RT=/data/nvme3/rt/$NAME; mkdir -p $R $RT
. /data/nvme3/shona-cpt/enroot/env.sh
exec enroot start -m /data/nvme3:/data/nvme3 \
  -m /data/nvme3/shona-cpt/cache/home:/home/ubuntu \
  -e HOME=/home/ubuntu -e HF_HOME=/data/nvme3/shona-cpt/hf_cache \
  -e VLLM_CACHE_ROOT=/data/nvme3/shona-cpt/cache/vllm -e TRITON_CACHE_DIR=/data/nvme3/shona-cpt/cache/triton \
  -e PYTHONPATH=/data/nvme3/dfly2/AngelSpec:$F/pydeps:$F/stubs \
  -e MC_STORE_MEMCPY=0 -e RAY_TMPDIR=$RT \
  -e VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3","quant_mode":"dynamic_token"}' \
  -e DSV41_TARGET_MODEL=/data/nvme3/lab/teachers/deepseek-v4.1-flash-1017 \
  -e DSV41_TRAIN_DATA=$DATA -e ANGELSPEC_RUN_DIR=$R \
  vllm bash -c "cd /data/nvme3/dfly2/AngelSpec && python3 -m angelspec.train_entry \
    --config configs/vllm_dsv41_flash_dfly2.yaml \
    training.training_num_gpus_per_node=4 \
    inference.inference_num_gpus=4 inference.inference_num_gpus_per_engine=4 \
    inference.inference_num_gpus_per_node=4 inference.vllm.tp_size=4 \
    inference.vllm.enable_multimodal=false \
    training.max_seq_length=12288 inference.vllm.extra_args.max_model_len=12288 \
    training.draft_init_path=/data/nvme3/dfly2/ft/init/release_train_layout \
    model.draft_model_config=/data/nvme3/dfly2/ft/init/draft_config_ft.json \
    $*"
