#!/bin/bash
# Copy the replication set (scripts, patches, configs, results, reports) into the git repo (branch measured-runs, folder measured/)
# /data/nvme3/h200-inference. No model weights, no large generated data (listed by sha256).
set -euo pipefail
N=/data/nvme3; R=$N/h200-inference/measured; D=$N/dfly2; F=$D/ft; T=$N/lab/teachers
mkdir -p $R
# DeepSeek V4.1 production serving (2 x TP4, W4A8, DSpark 3)
mkdir -p $R/deepseek-v41/patches
rsync -a --delete $N/inference/deploy/ $R/deepseek-v41/deploy/
rsync -a --delete $T/patches/humming_tune $T/patches/prefill_interval $R/deepseek-v41/patches/
cp $T/quality_v41.py $T/bench_latency2_v41.sh $R/deepseek-v41/
cp $N/inference/notes-infrence.md $R/notes-inference.md
# DFly2 served through vLLM (port of AngelSpec DFly2 to vLLM 0.32.0)
mkdir -p $R/dfly2/serving
rsync -a --delete --exclude __pycache__ $D/patches/ $R/dfly2/serving/patches/
rsync -a --delete --exclude __pycache__ $D/pkg/ $R/dfly2/serving/pkg/
rsync -a --delete $D/tools/ $R/dfly2/serving/tools/
# DFly2 fine-tune on V4.1 thinking-mode replies
mkdir -p $R/dfly2/finetune/support
cp $F/*.py $F/*.sh $R/dfly2/finetune/
cp $F/init/draft_config_ft.json $R/dfly2/finetune/
rsync -a --delete $F/stubs/ $R/dfly2/finetune/support/stubs/
rsync -a --delete $F/pydeps/angelspec-0.0.0+local.dist-info/ $R/dfly2/finetune/support/angelspec-0.0.0+local.dist-info/
ls $F/pydeps | grep -E "\.dist-info$" | grep -v angelspec | sed 's/\.dist-info$//' | sort > $R/dfly2/finetune/support/pydeps-versions.txt
[ -f $F/runs/ft_v1/dfly2_dsv41_flash/config.yaml ] && cp $F/runs/ft_v1/dfly2_dsv41_flash/config.yaml $R/dfly2/finetune/train_ft_v1_resolved_config.yaml
# Results
mkdir -p $R/dfly2/results/logs
rsync -a --delete $D/results/ $R/dfly2/results/data/
rsync -a $D/logs/ $R/dfly2/results/logs/serving/
for f in gen.log chain.log eval.log capture_smoke.log; do [ -f $F/logs/$f ] && grep -a -v CUDACachingAllocator $F/logs/$f > $R/dfly2/results/logs/$f; done
for f in train_ft_v1.log train_smoke.log; do [ -f $F/logs/$f ] && tr '\r' '\n' < $F/logs/$f | grep -a -v -E "CUDACachingAllocator|Loading safetensors checkpoint shards|Prefetching checkpoint" > $R/dfly2/results/logs/$f; done
# per-step training metrics
tr '\r' '\n' < $F/logs/train_ft_v1.log | grep -a -o "| *[0-9]*/364 .*epoch=[0-9]/2" \
 | sed -E 's/^\| *([0-9]+)\/364.*loss=([0-9.]+), acc=([0-9.]+), acc_len=([0-9.]+), thru=([0-9.]+), acc_len_tgt=([0-9.]+).*epoch=([0-9])\/2/\1,\2,\3,\4,\6,\7/' \
 | awk -F, '!s[$1]++' | (echo "step,loss,acc_first_draft,acc_len,acc_len_tgt,epoch"; cat) > $R/dfly2/results/train_ft_v1_metrics.csv
# large generated data: checksums only
( cd $F/data && sha256sum *.json *.jsonl ) > $R/dfly2/finetune/data_sha256.txt
( cd $F/data && wc -l *.jsonl ) > $R/dfly2/finetune/data_line_counts.txt
# reports
mkdir -p $R/reports; cp $N/inference/report/*.pdf $R/reports/
echo "synced $(date -Is)"
