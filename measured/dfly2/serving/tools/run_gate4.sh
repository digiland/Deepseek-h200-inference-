#!/bin/bash
# Gate 4 driver: for each server config, start it on GPUs 4-7 (port 8190), run the temp-0
# acceptance probe, the quality probe and the sustained ShareGPT benchmark at 20/40/80 users
# (V41_NMULT=16, same fresh seed base for every config), then stop it.
# usage: run_gate4.sh "<tag> <MODE> <NSPEC> <REJ>" ...
set -uo pipefail
W=/data/nvme3/dfly2; PORT=8190; LOG=$W/logs/gate4.log
export V41_NMULT=16 V41_SEED_BASE=52000
exec >> "$LOG" 2>&1
wait_up() {
  for _ in $(seq 1 120); do
    curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && return 0
    grep -q "Traceback" $W/logs/serve-$PORT.log && return 1
    sleep 10
  done; return 1
}
stop_server() {
  for p in $(pgrep -f "vllm.entrypoints.openai.api_server"); do
    if tr '\0' ' ' < /proc/$p/cmdline | grep -q -- "--port $PORT" &&
       tr '\0' '\n' < /proc/$p/environ | grep -q "CUDA_VISIBLE_DEVICES=4,5,6,7"; then
      kill -TERM -"$(ps -o pgid= -p $p | tr -d ' ')"
    fi
  done
  for _ in $(seq 1 60); do
    [ "$(nvidia-smi -i 4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 2000 ] && return
    sleep 5
  done
  echo "WARN: GPUs 4-7 not free after stop"
}
for cfg in "$@"; do
  read -r tag mode nspec rej <<< "$cfg"
  echo "[$(date -Is)] === $tag (MODE=$mode NSPEC=$nspec REJ=$rej)"
  if ! curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
    MODE=$mode NSPEC=$nspec REJ=$rej PORT=$PORT setsid nohup bash $W/tools/serve_v41_gpu47.sh >/dev/null 2>&1 &
    sleep 30
  fi
  if ! wait_up; then echo "server failed for $tag"; stop_server; continue; fi
  python3 $W/tools/accept_probe.py $PORT "$tag-t0-short" 80 0 short
  python3 /data/nvme3/lab/teachers/quality_v41.py $PORT "$tag" $W/results/quality | tail -1
  bash $W/tools/bench_gpu47.sh "$tag" $PORT sharegpt 20 40 80
  stop_server
done
echo "[$(date -Is)] gate4 done"
