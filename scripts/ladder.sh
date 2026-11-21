#!/usr/bin/env bash
# Run serving profiles back to back and compare them.
#
#   scripts/ladder.sh                       # LADDER=core (default) or LADDER=extended
#   scripts/ladder.sh vllm:baseline 'vllm:max-nospec;NODE_SEQS=2048' sglang:dpa4
#
# Entry syntax: <engine>:<profile>[;KEY=VALUE...] (';' because values may hold JSON commas). The KEY=VALUEs are env overrides for
# that entry only (any serve/*/launch.sh knob, or H200OPT_* plugin knobs). They are also
# part of the result label.
#
# For each entry: start the server (Docker, or natively with RUNNER=native), wait for
# /health, optionally start the prefix-affinity router (DP profiles, ROUTER=1), run the
# quality gate, run bench/run_sweep.sh, stop. Results: results/<label>/, comparisons
# against BASE_LABEL (default: the first entry): results/compare_<base>_vs_<label>.md
#
# Env:
#   RUNNER         docker (default) | native
#   MODEL_DIR      host checkpoint dir (docker) / MODEL_PATH (native)
#   SHAREGPT_PATH  host path to ShareGPT_V3_unfiltered_cleaned_split.json (headline scenarios)
#   SCENARIOS      default "sharegpt multiturn decode mm"
#   ROUTER         1 (default): DP profiles are benchmarked through serve/router/affinity_router.py
#   QUALITY        1 (default): run bench/quality_gate.py per profile (GSM8K n=QUALITY_N, 400)
#   QUALITY_REF    label whose quality.json later profiles are compared to (default: the first
#                  profile that ran the gate). For the FP8 check use e.g. QUALITY_REF=vllm-dep4-mega-spec
#   READY_TIMEOUT  seconds to wait for the server (default 7200)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUNNER=${RUNNER:-docker}
PORT=${PORT:-8000}
ROUTER_PORT=${ROUTER_PORT:-8080}
READY_TIMEOUT=${READY_TIMEOUT:-7200}
SCENARIOS=${SCENARIOS:-"sharegpt multiturn decode mm"}
ROUTER=${ROUTER:-1}
QUALITY=${QUALITY:-1}
QUALITY_N=${QUALITY_N:-400}
SERVED_NAME=${SERVED_NAME:-deepseek-ai/DeepSeek-V4.1-Flash}

CORE=(vllm:baseline vllm:baseline-spec vllm:tp4-tuned-spec vllm:dep4-mega-spec vllm:max
      vllm:max-nospec 'vllm:max-nospec;NODE_SEQS=2048')
EXTENDED=(vllm:baseline vllm:baseline-spec vllm:tp4-tuned vllm:tp4-tuned-spec
          vllm:dep4 'vllm:dep4;EXTRA_ARGS=--all2all-backend alltoall_batched'
          vllm:dep4-mega vllm:dep4-mega-spec 'vllm:dep4-mega-spec;EPLB=1'
          'vllm:dep4-mega-spec;SPEC_SCHEDULE=[[1,64,5],[65,128,3],[129,1024,1]]'
          vllm:max vllm:max-nospec 'vllm:max-nospec;NODE_SEQS=2048' 'vllm:max-nospec;NODE_SEQS=4096'
          sglang:baseline sglang:dpa4 sglang:dpa4-nospec)
LADDER=("$@")
if (( ${#LADDER[@]} == 0 )); then
  if [[ "${LADDER_SET:-core}" == extended || "${LADDER:-}" == extended ]]; then LADDER=("${EXTENDED[@]}"); else LADDER=("${CORE[@]}"); fi
fi
if [[ -z "${SHAREGPT_PATH:-}" ]]; then
  echo "note: SHAREGPT_PATH unset -> dropping sharegpt/multiturn (headline numbers need them)"
  SCENARIOS=$(echo " $SCENARIOS " | sed 's/ sharegpt / /; s/ multiturn / /')
fi

is_dp() { [[ "$1" == vllm && "$2" == dep4* || "$1" == vllm && "$2" == max* ]]; }

wait_ready() {
  local t0=$SECONDS
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && \
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; do
    (( SECONDS - t0 > READY_TIMEOUT )) && return 1
    if [[ "$RUNNER" == docker ]] && ! docker ps --format '{{.Names}}' | grep -qx "dsv41-$1"; then
      echo "container dsv41-$1 exited"; return 1
    fi
    sleep 15
  done
  echo "ready after $(( SECONDS - t0 ))s"
}

start() {  # engine profile
  if [[ "$RUNNER" == docker ]]; then PORT=$PORT "$ROOT/docker/run.sh" up "$1" "$2"
  else
    nohup "$ROOT/serve/$1/launch.sh" "$2" > "$ROOT/results/server-$LABEL.log" 2>&1 &
    echo $! > "$ROOT/results/.server.pid"
  fi
}

stop() {  # engine
  if [[ "$RUNNER" == docker ]]; then
    docker logs "dsv41-$1" > "$ROOT/results/server-$LABEL.log" 2>&1 || true
    "$ROOT/docker/run.sh" down "$1" >/dev/null 2>&1 || true
    docker rm -f dsv41-router >/dev/null 2>&1 || true
  else
    kill "$(cat "$ROOT/results/.server.pid")" 2>/dev/null || true
    [[ -f "$ROOT/results/.router.pid" ]] && kill "$(cat "$ROOT/results/.router.pid")" 2>/dev/null
    sleep 20; pkill -f "vllm serve|sglang.launch_server" 2>/dev/null || true
  fi
  sleep 10
}

start_router() {
  if [[ "$RUNNER" == docker ]]; then
    docker run -d --rm --name dsv41-router --network host --entrypoint python3 h200/vllm:latest \
      /opt/h200/serve/router/affinity_router.py --upstream "http://127.0.0.1:$PORT" \
      --port "$ROUTER_PORT" --dp 4 >/dev/null
  else
    nohup python3 "$ROOT/serve/router/affinity_router.py" --upstream "http://127.0.0.1:$PORT" \
      --port "$ROUTER_PORT" --dp 4 > "$ROOT/results/router-$LABEL.log" 2>&1 &
    echo $! > "$ROOT/results/.router.pid"
  fi
  sleep 3
}

in_client() {  # run a command from the repo root, in the client image (docker) or on the host
  if [[ "$RUNNER" == docker ]]; then
    local mounts=(-v "$MODEL_DIR":/model:ro -v "$ROOT/results":/opt/h200/results
                  -v "$ROOT/third_party/vllm/benchmarks/multi_turn":/opt/h200/third_party/vllm/benchmarks/multi_turn:ro)
    local envs=(-e TOKENIZER=/model -e MODEL="$SERVED_NAME" -e PORT="$BENCH_PORT")
    if [[ -n "${SHAREGPT_PATH:-}" ]]; then
      mounts+=(-v "$SHAREGPT_PATH":/data/sharegpt.json:ro); envs+=(-e SHAREGPT_PATH=/data/sharegpt.json)
    fi
    for v in CONCURRENCIES REPEATS PROMPTS_MULT RANGE_RATIO; do [[ -n "${!v:-}" ]] && envs+=(-e "$v=${!v}"); done
    docker run --rm --network host --entrypoint bash "${mounts[@]}" "${envs[@]}" \
      -w /opt/h200 h200/vllm:latest -c "$*"
  else
    (cd "$ROOT" && PORT=$BENCH_PORT TOKENIZER=${MODEL_PATH:-} MODEL=$SERVED_NAME bash -c "$*")
  fi
}

mkdir -p "$ROOT/results"
if [[ "$RUNNER" == docker ]]; then
  : "${MODEL_DIR:?set MODEL_DIR}"
  docker image inspect h200/vllm:latest >/dev/null 2>&1 || "$ROOT/docker/run.sh" build vllm
fi

LABELS=()
REF_QUALITY=${QUALITY_REF:-}
for entry in "${LADDER[@]}"; do
  spec=${entry%%;*}; overrides=""; [[ "$entry" == *\;* ]] && overrides=${entry#*;}
  engine=${spec%%:*}; profile=${spec#*:}
  LABEL="$engine-$profile"
  saved_pass_env=${PASS_ENV:-}
  declare -a set_vars=()
  if [[ -n "$overrides" ]]; then
    IFS=';' read -r -a kvs <<< "$overrides"
    for kv in "${kvs[@]}"; do
      export "${kv%%=*}=${kv#*=}"; set_vars+=("${kv%%=*}")
      LABEL+="+$(echo "${kv%%=*}=${kv#*=}" | tr -c 'A-Za-z0-9=_.\n-' '_' | cut -c1-40)"
    done
    export PASS_ENV="$saved_pass_env ${set_vars[*]}"
  fi
  echo "################ $LABEL ################"
  if [[ "$RUNNER" == docker && "$engine" == sglang ]]; then
    docker image inspect h200/sglang:latest >/dev/null 2>&1 || "$ROOT/docker/run.sh" build sglang
  fi
  start "$engine" "$profile"
  if wait_ready "$engine"; then
    BENCH_PORT=$PORT
    if [[ "$ROUTER" == 1 ]] && is_dp "$engine" "$profile"; then start_router; BENCH_PORT=$ROUTER_PORT; fi
    if [[ "$QUALITY" == 1 ]]; then
      ref_arg=""
      [[ -n "$REF_QUALITY" && "$REF_QUALITY" != "$LABEL" && -f "$ROOT/results/$REF_QUALITY/quality.json" ]] \
        && ref_arg="--ref results/$REF_QUALITY/quality.json"
      in_client "python3 bench/quality_gate.py --gsm8k-n $QUALITY_N --save results/$LABEL/quality.json $ref_arg" \
        | tee "$ROOT/results/quality-$LABEL.txt" \
        || echo "!!! $LABEL: quality gate FAILED (see results/quality-$LABEL.txt)"
      [[ -z "$REF_QUALITY" && -f "$ROOT/results/$LABEL/quality.json" ]] && REF_QUALITY=$LABEL
    fi
    in_client "bench/run_sweep.sh '$LABEL' $SCENARIOS"
    LABELS+=("$LABEL")
  else
    echo "!!! $LABEL failed to start; see results/server-$LABEL.log"
  fi
  stop "$engine"
  for v in "${set_vars[@]}"; do unset "$v"; done
  export PASS_ENV=$saved_pass_env
done

base=${BASE_LABEL:-${LABELS[0]:-}}
for l in "${LABELS[@]}"; do
  [[ "$l" == "$base" ]] && continue
  python3 "$ROOT/bench/summarize.py" "$ROOT/results/$base" "$ROOT/results/$l" > /dev/null \
    && cp "$ROOT/results/$l/compare_vs_$base.md" "$ROOT/results/compare_${base}_vs_${l}.md"
done
echo "done. per-run: results/<label>/summary.md, quality: results/quality-*.txt, comparisons: results/compare_*.md"
