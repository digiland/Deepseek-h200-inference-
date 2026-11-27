#!/bin/bash
# DFly2 vs DSpark full benchmark matrix, DeepSeek-V4.1-Flash, 8 x H200.
# Same protocol as run_final_v41.sh: ShareGPT, T=0.6, 16 x users requests per rung, fresh seeds,
# W4A8 + Humming 2 CTAs/SM + prompts every 8th step. Results: results/bench/results.csv.
#   Layout D (2 copies x TP4, GPUs 0-3 port 8180 + GPUs 4-7 port 8181), users per copy 20 40 80 100
#   Layout S (1 copy x TP8, GPUs 0-7, port 8182), users 40 80 160 200
#   High load H (2 x TP4), users per copy 256
#   Thinking: default (template on) or off (chat_template_kwargs thinking false).
set -uo pipefail
W=/data/nvme3/dfly2; T=/data/nvme3/lab/teachers
LOG=$W/logs/matrix.log; exec >> "$LOG" 2>&1
export V41_NMULT=16
OFF='{"thinking": false, "enable_thinking": false}'
gpu_max() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1; }
waitfree() { for _ in $(seq 1 60); do [ "$(gpu_max)" -lt 4000 ] && return 0; sleep 5; done; echo "WARN: GPUs still busy"; }
PIDS=()
start() {  # port gpus tp mode nspec rej
  local port=$1
  PORT=$1 GPUS=$2 TP=$3 MODE=$4 NSPEC=$5 REJ=$6 DRAFT_SAMPLE=probabilistic setsid bash $W/tools/serve_v41_any.sh &
  PIDS+=($!)
  for _ in $(seq 1 240); do
    curl -sf http://127.0.0.1:$port/health >/dev/null 2>&1 && { echo "server $port up $(date -Is)"; return 0; }
    kill -0 ${PIDS[-1]} 2>/dev/null || { echo "SERVER FAILED $port"; tail -5 $W/logs/serve-$port.log; return 1; }
    sleep 15; done; echo "SERVER TIMEOUT $port"; return 1
}
stopall() { for p in "${PIDS[@]}"; do kill -- -$p 2>/dev/null; kill $p 2>/dev/null; done; PIDS=(); sleep 10; waitfree; }
quality() { python3 $T/quality_v41.py $1 "$2" $W/results/quality || echo "WARN quality failed"; }
dual() {  # tag mode nspec rej think concs...
  local tag=$1 mode=$2 nspec=$3 rej=$4 think=$5; shift 5
  echo "================ $(date -Is) matrix=$tag (2 x TP4) $mode n=$nspec rej=$rej think=$think"
  if start 8180 0,1,2,3 4 $mode $nspec $rej && start 8181 4,5,6,7 4 $mode $nspec $rej; then
    quality 8180 "${tag}A"
    local extra=""; [ "$think" = off ] && extra="$OFF"
    for c in "$@"; do
      BENCH_EXTRA="$extra" V41_SEED_BASE=61000 bash $W/tools/bench_any.sh "${tag}A" 8180 sharegpt $c & local b1=$!
      BENCH_EXTRA="$extra" V41_SEED_BASE=62000 bash $W/tools/bench_any.sh "${tag}B" 8181 sharegpt $c & local b2=$!
      wait $b1 $b2
    done
  fi
  stopall
}
single() {  # tag mode nspec rej think concs...
  local tag=$1 mode=$2 nspec=$3 rej=$4 think=$5; shift 5
  echo "================ $(date -Is) matrix=$tag (1 x TP8) $mode n=$nspec rej=$rej think=$think"
  if start 8182 0,1,2,3,4,5,6,7 8 $mode $nspec $rej; then
    quality 8182 "$tag"
    local extra=""; [ "$think" = off ] && extra="$OFF"
    BENCH_EXTRA="$extra" V41_SEED_BASE=63000 bash $W/tools/bench_any.sh "$tag" 8182 sharegpt "$@"
  fi
  stopall
}
mkdir -p $W/results/quality
echo "================ $(date -Is) matrix start, max GPU mem $(gpu_max) MiB"
waitfree
# Layout D, thinking default (the benchmark default), then thinking off
dual M1_dspark3_think dspark 3 block   default 20 40 80 100
dual M2_dfly3_think   dfly2  3 standard default 20 40 80 100
dual M3_dfly4_think   dfly2  4 standard default 20 40 80 100
dual M4_dspark3_off   dspark 3 block   off     20 40 80 100
dual M5_dfly3_off     dfly2  3 standard off     20 40 80 100
# Layout S, thinking default
single M6_dspark3_tp8 dspark 3 block   default 40 80 160 200
single M7_dfly3_tp8   dfly2  3 standard default 40 80 160 200
# High load, 2 x TP4, thinking default
dual M8_dspark3_high  dspark 3 block   default 256
dual M9_dfly3_high    dfly2  3 standard default 256
echo "================ $(date -Is) matrix done"
