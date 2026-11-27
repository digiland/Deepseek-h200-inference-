#!/bin/bash
# After ft_v1 training completes: export the final drafter to serving format, then run the eval.
F=/data/nvme3/dfly2/ft; L=$F/logs/train_ft_v1.log; OUT=$F/served/DFly2-ft-v1
until grep -a -q -E "Training completed|Traceback" $L; do sleep 30; done
grep -a -q "Traceback" $L && { echo "training failed, no eval"; exit 1; }
for _ in $(seq 1 120); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 4000 ] && break; sleep 10; done
. /data/nvme3/shona-cpt/enroot/env.sh
enroot start -m /data/nvme3:/data/nvme3 -e CUDA_VISIBLE_DEVICES= vllm bash -c "cd $F && PYTHONPATH=$F/pydeps python3 train_to_release.py runs/ft_v1/dfly2_dsv41_flash/hf_final /data/nvme3/dfly2/served/DS-V4.1-Flash-DFly2-B6 $OUT" || { echo "export failed"; exit 1; }
echo "exported $OUT $(date -Is)"
exec $F/run_ft_eval.sh $OUT
