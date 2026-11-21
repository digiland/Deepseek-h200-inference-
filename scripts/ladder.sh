#!/usr/bin/env bash
# Run the optimization ladder end to end and produce a comparison table.
#
#   scripts/ladder.sh [engine:profile ...]
#
# Default ladder (vLLM): baseline tp4-spec dep4 dep4-mega dep4-mega-spec max max-nospec
# Example mixing engines:
#   scripts/ladder.sh vllm:baseline vllm:max sglang:baseline sglang:dpa4
#
# For each entry: start the server (Docker, or natively with RUNNER=native), wait
# for /health, run bench/run_sweep.sh, scrape /metrics, stop the server.
# Results: results/<engine>-<profile>/, comparisons: results/compare_*.md
#
# Env:
#   RUNNER      docker (default) | native
#   MODEL_DIR   host checkpoint dir (docker) / MODEL_PATH (native)
#   SCENARIOS   scenarios passed to run_sweep.sh (default: "decode chat sharegpt prefix")
#   SHAREGPT_PATH   host path to ShareGPT_V3_unfiltered_cleaned_split.json (for "sharegpt")
#   READY_TIMEOUT   seconds to wait for the server (default 7200: first load streams ~475 GB)
#   BASE_LABEL  label to compare against (default: first entry of the ladder)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUNNER=${RUNNER:-docker}
PORT=${PORT:-8000}
READY_TIMEOUT=${READY_TIMEOUT:-7200}
SCENARIOS=${SCENARIOS:-"decode chat sharegpt prefix"}
LADDER=("$@")
if (( ${#LADDER[@]} == 0 )); then
  LADDER=(vllm:baseline vllm:tp4-spec vllm:dep4 vllm:dep4-mega vllm:dep4-mega-spec vllm:max vllm:max-nospec)
fi
[[ " $SCENARIOS " == *" sharegpt "* && -z "${SHAREGPT_PATH:-}" ]] && {
  echo "note: SHAREGPT_PATH unset, dropping the sharegpt scenario"; SCENARIOS=${SCENARIOS//sharegpt/}; }

wait_ready() {
  local t0=$SECONDS
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && \
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; do
    if (( SECONDS - t0 > READY_TIMEOUT )); then return 1; fi
    if [[ "$RUNNER" == docker ]] && ! docker ps --format '{{.Names}}' | grep -qx "dsv41-$1"; then
      echo "container dsv41-$1 exited"; return 1
    fi
    sleep 15
  done
  echo "ready after $(( SECONDS - t0 ))s"
}

start() {  # engine profile
  if [[ "$RUNNER" == docker ]]; then
    PORT=$PORT "$ROOT/docker/run.sh" up "$1" "$2"
  else
    nohup "$ROOT/serve/$1/launch.sh" "$2" > "$ROOT/results/server-$1-$2.log" 2>&1 &
    echo $! > "$ROOT/results/.server.pid"
  fi
}

stop() {  # engine
  if [[ "$RUNNER" == docker ]]; then
    docker logs "dsv41-$1" > "$ROOT/results/server-$1-${2}.log" 2>&1 || true
    "$ROOT/docker/run.sh" down "$1" >/dev/null 2>&1 || true
  else
    kill "$(cat "$ROOT/results/.server.pid")" 2>/dev/null || true
    sleep 20; pkill -f "vllm serve|sglang.launch_server" 2>/dev/null || true
  fi
  sleep 10
}

bench() {  # label
  if [[ "$RUNNER" == docker ]]; then
    local mounts=(-v "$MODEL_DIR":/model:ro -v "$ROOT/results":/opt/h200/results)
    local sg=()
    if [[ -n "${SHAREGPT_PATH:-}" ]]; then
      mounts+=(-v "$SHAREGPT_PATH":/data/sharegpt.json:ro); sg=(-e SHAREGPT_PATH=/data/sharegpt.json)
    fi
    docker run --rm --network host --entrypoint /opt/h200/bench/run_sweep.sh \
      "${mounts[@]}" "${sg[@]}" -e TOKENIZER=/model -e PORT="$PORT" -e MODEL="${SERVED_NAME:-deepseek-ai/DeepSeek-V4.1-Flash}" \
      ${CONCURRENCIES:+-e CONCURRENCIES="$CONCURRENCIES"} \
      h200/vllm:latest "$1" $SCENARIOS
  else
    PORT=$PORT TOKENIZER=${MODEL_PATH:-} "$ROOT/bench/run_sweep.sh" "$1" $SCENARIOS
  fi
}

mkdir -p "$ROOT/results"
if [[ "$RUNNER" == docker ]]; then
  : "${MODEL_DIR:?set MODEL_DIR}"
  docker image inspect h200/vllm:latest >/dev/null 2>&1 || "$ROOT/docker/run.sh" build vllm
fi

LABELS=()
for entry in "${LADDER[@]}"; do
  engine=${entry%%:*}; profile=${entry#*:}; label="$engine-$profile"
  echo "################ $label ################"
  if [[ "$RUNNER" == docker && "$engine" == sglang ]]; then
    docker image inspect h200/sglang:latest >/dev/null 2>&1 || "$ROOT/docker/run.sh" build sglang
  fi
  start "$engine" "$profile"
  if wait_ready "$engine"; then
    bench "$label"
    LABELS+=("$label")
  else
    echo "!!! $label failed to start; see results/server-$engine-$profile.log"
  fi
  stop "$engine" "$profile"
done

base=${BASE_LABEL:-${LABELS[0]:-}}
for l in "${LABELS[@]}"; do
  [[ "$l" == "$base" ]] && continue
  python3 "$ROOT/bench/summarize.py" "$ROOT/results/$base" "$ROOT/results/$l" > /dev/null
  cp "$ROOT/results/$l/compare_vs_$base.md" "$ROOT/results/compare_${base}_vs_${l}.md" 2>/dev/null
done
echo "done. per-run: results/<label>/summary.md, comparisons: results/compare_*.md"
