#!/bin/bash
# Evaluate the fine-tuned DFly2 drafter against the released DFly2 and DSpark, all on the
# held-out 15% of ShareGPT (never used for fine-tuning prompts). Same protocol as run_matrix.sh:
# 2 copies x TP4, thinking on (template default), T=0.6, 16 x users requests per rung, fresh seeds.
# Per setup: quality probe (50 questions), acceptance probe (160 prompts, T=0.6), benchmark.
# usage: run_ft_eval.sh <fine-tuned served dir>
set -uo pipefail
W=/data/nvme3/dfly2; T=/data/nvme3/lab/teachers; F=$W/ft
FT=${1:?usage: run_ft_eval.sh <fine-tuned served dir>}
LOG=$F/logs/eval.log; exec >> "$LOG" 2>&1
export V41_NMULT=16 SHAREGPT=$F/data/sharegpt_heldout.json
gpu_max() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1; }
waitfree() { for _ in $(seq 1 60); do [ "$(gpu_max)" -lt 4000 ] && return 0; sleep 5; done; echo "WARN: GPUs still busy"; }
PIDS=()
start() {  # port gpus mode nspec rej model
  local port=$1
  DFLY2_MODEL=$6 PORT=$1 GPUS=$2 TP=4 MODE=$3 NSPEC=$4 REJ=$5 DRAFT_SAMPLE=probabilistic setsid bash $W/tools/serve_v41_any.sh &
  PIDS+=($!)
  for _ in $(seq 1 240); do
    curl -sf http://127.0.0.1:$port/health >/dev/null 2>&1 && { echo "server $port up $(date -Is)"; return 0; }
    kill -0 ${PIDS[-1]} 2>/dev/null || { echo "SERVER FAILED $port"; tail -5 $W/logs/serve-$port.log; return 1; }
    sleep 15; done; echo "SERVER TIMEOUT $port"; return 1
}
stopall() { for p in "${PIDS[@]}"; do kill -- -$p 2>/dev/null; kill $p 2>/dev/null; done; PIDS=(); sleep 10; waitfree; }
dual() {  # tag mode nspec rej model concs...
  local tag=$1 mode=$2 nspec=$3 rej=$4 model=$5; shift 5
  echo "================ $(date -Is) eval=$tag (2 x TP4) $mode n=$nspec rej=$rej model=$model"
  if start 8180 0,1,2,3 $mode $nspec $rej $model && start 8181 4,5,6,7 $mode $nspec $rej $model; then
    python3 $T/quality_v41.py 8180 "${tag}A" $W/results/quality || echo "WARN quality failed"
    PROBE_TEMP=0.6 PROBE_TOP_P=0.95 python3 $W/tools/accept_probe.py 8180 "$tag" 160 d short
    for c in "$@"; do
      V41_SEED_BASE=71000 bash $W/tools/bench_any.sh "${tag}A" 8180 sharegpt $c & local b1=$!
      V41_SEED_BASE=72000 bash $W/tools/bench_any.sh "${tag}B" 8181 sharegpt $c & local b2=$!
      wait $b1 $b2
    done
  fi
  stopall
}
echo "================ $(date -Is) eval start, held-out prompts $SHAREGPT"
waitfree
REL=$W/served/DS-V4.1-Flash-DFly2-B6
dual E1_dspark3_ho   dspark 3 block    none 20 40 80 100
dual E2_dfly3rel_ho  dfly2  3 standard $REL 20 40 80 100
dual E3_dfly3ft_ho   dfly2  3 standard $FT  20 40 80 100
dual E4_dfly4ft_ho   dfly2  4 standard $FT  20 40 80 100
echo "================ $(date -Is) eval done"
