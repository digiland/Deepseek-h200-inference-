#!/bin/bash
# Stage 1 of the DFly2 fine-tune: V4.1 thinking-mode replies for 20,000 ShareGPT training prompts.
. /data/nvme3/shona-cpt/enroot/env.sh
exec enroot start -m /data/nvme3:/data/nvme3 -e MAX_TOKENS=8192 vllm bash -c \
  'cd /data/nvme3/dfly2/ft && python3 gen_replies.py data/train_prompts.jsonl data/replies_v1.jsonl 20000 8180 8181'
