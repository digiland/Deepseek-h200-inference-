#!/usr/bin/env bash
# Engine-neutral throughput sweep against any OpenAI-compatible server
# (vLLM, SGLang, ...). Uses `vllm bench serve` (and vLLM's multi-turn benchmark)
# as the one load generator, so every engine is measured by the same client.
#
# Usage:
#   bench/run_sweep.sh <label> [scenario ...]
#
#   label     tag such as "vllm-baseline"; results go to results/<label>/
#   scenario  any of the names below (default: sharegpt multiturn decode mm)
#
# Headline (real text, so DSpark acceptance, Engram hashes and the indexer behave realistically):
#   sharegpt   ShareGPT prompts, natural outputs capped at 1k tokens
#   multiturn  ShareGPT conversations replayed turn by turn (vLLM benchmarks/multi_turn),
#              conversations interleaved, so DP prefix-cache affinity matters
#   mm         chat endpoint, 1 random image + 512 text tokens, 256 out (multimodal path)
# Synthetic (kernel-level A/B only: `random` prompts are consecutive token ids):
#   smoke      512/128, c=8
#   decode     1k/1k     chat 4k/1k     long 32k/1k     prefill 8k/64
#   prefix     16 shared 4k prefixes + 512 unique suffix, 512 out
#
# Env:
#   HOST/PORT          server address (default 127.0.0.1:8000); point PORT at the
#                      affinity router (8080) to measure routed DP serving
#   MODEL / TOKENIZER  served name (default: first /v1/models entry) / tokenizer path
#   SHAREGPT_PATH      ShareGPT_V3_unfiltered_cleaned_split.json (sharegpt, multiturn)
#   CONCURRENCIES      override concurrency list, e.g. "256 1024"
#   PROMPTS_MULT       prompts per concurrency level (default 6 x c; min 64, max MAX_PROMPTS=8192)
#   RANGE_RATIO        length jitter for synthetic scenarios (default 0.3), avoids phase-locked waves
#   REPEATS            repeat each point (default 1; use 3 for the final numbers)
#   MT_DIR             vLLM benchmarks/multi_turn dir (default third_party/vllm/benchmarks/multi_turn)
#   BENCH_PY           python with vllm installed (default python3)
set -euo pipefail

LABEL=${1:?usage: run_sweep.sh <label> [scenario ...]}
shift || true
SCENARIOS=${*:-"sharegpt multiturn decode mm"}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
BENCH_PY=${BENCH_PY:-python3}
PROMPTS_MULT=${PROMPTS_MULT:-6}
MAX_PROMPTS=${MAX_PROMPTS:-8192}
RANGE_RATIO=${RANGE_RATIO:-0.3}
REPEATS=${REPEATS:-1}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MT_DIR=${MT_DIR:-$ROOT/third_party/vllm/benchmarks/multi_turn}
OUT=${OUT:-$ROOT/results/$LABEL}
BASE="http://$HOST:$PORT"
mkdir -p "$OUT"

if [[ -z "${MODEL:-}" ]]; then
  MODEL=$("$BENCH_PY" -c 'import json,sys,urllib.request; print(json.load(urllib.request.urlopen(sys.argv[1]))["data"][0]["id"])' "$BASE/v1/models")
fi
TOKENIZER=${TOKENIZER:-${MODEL_PATH:-$MODEL}}

{
  echo "date: $(date -Is)"; echo "label: $LABEL"; echo "model: $MODEL"; echo "endpoint: $BASE"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,clocks.max.sm --format=csv,noheader 2>/dev/null || true
} > "$OUT/env.txt"
[[ -f "$ROOT/results/.last_server_cmd" ]] && cp "$ROOT/results/.last_server_cmd" "$OUT/server_cmd.txt" || true

declare -A CONCS=(
  [smoke]="8" [decode]="128 256 512 1024 2048" [chat]="128 256 512 1024" [long]="16 32 64 128"
  [prefill]="32 64 128" [sharegpt]="128 256 512 1024 2048" [prefix]="64 256 512" [mm]="32 64 128 256"
  [multiturn]="128 256 512"
)
RJ="--random-range-ratio $RANGE_RATIO"
dataset_args() {
  case "$1" in
    smoke)   echo --dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos ;;
    decode)  echo --dataset-name random --random-input-len 1024 --random-output-len 1024 --ignore-eos $RJ ;;
    chat)    echo --dataset-name random --random-input-len 4096 --random-output-len 1024 --ignore-eos $RJ ;;
    long)    echo --dataset-name random --random-input-len 32768 --random-output-len 1024 --ignore-eos $RJ ;;
    prefill) echo --dataset-name random --random-input-len 8192 --random-output-len 64 --ignore-eos $RJ ;;
    sharegpt)
      : "${SHAREGPT_PATH:?set SHAREGPT_PATH to ShareGPT_V3_unfiltered_cleaned_split.json}"
      echo --dataset-name sharegpt --dataset-path "$SHAREGPT_PATH" --sharegpt-output-len 1024 ;;
    prefix)  echo --dataset-name prefix_repetition --prefix-repetition-prefix-len 4096 \
                  --prefix-repetition-suffix-len 512 --prefix-repetition-num-prefixes 16 \
                  --prefix-repetition-output-len 512 --ignore-eos ;;
    mm)      echo --backend openai-chat --endpoint /v1/chat/completions \
                  --dataset-name random-mm --random-input-len 512 --random-output-len 256 \
                  --random-mm-base-items-per-request 1 --ignore-eos $RJ ;;
    *) echo "unknown scenario $1" >&2; return 1 ;;
  esac
}

scrape() {  # stdlib only: the client image may not ship curl
  "$BENCH_PY" -c 'import sys,urllib.request; sys.stdout.write(urllib.request.urlopen(sys.argv[1], timeout=30).read().decode())' \
    "$BASE/metrics" > "$1" 2>/dev/null || true
}

run_serve_bench() {  # scenario concurrency name
  local sc=$1 c=$2 name=$3 dargs backend_args n
  dargs=$(dataset_args "$sc")
  backend_args="--backend openai --endpoint /v1/completions"
  [[ "$dargs" == *"--backend"* ]] && backend_args=""
  n=$(( c * PROMPTS_MULT )); (( n < 64 )) && n=64; (( n > MAX_PROMPTS )) && n=$MAX_PROMPTS
  echo "=== [$LABEL] $name prompts=$n ==="
  # shellcheck disable=SC2086
  "$BENCH_PY" -m vllm.entrypoints.cli.main bench serve $backend_args \
    --base-url "$BASE" --model "$MODEL" --tokenizer "$TOKENIZER" --trust-remote-code \
    $dargs --num-prompts "$n" --max-concurrency "$c" --request-rate inf \
    --num-warmups "$(( c < 32 ? c : 32 ))" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$OUT" --result-filename "$name.json" \
    2>&1 | tee "$OUT/$name.log" || echo "!!! $name failed (see log)"
}

run_multiturn() {  # concurrency name
  local c=$1 name=$2 conv="$ROOT/results/.cache/sharegpt_conv_2000.json"
  : "${SHAREGPT_PATH:?set SHAREGPT_PATH for multiturn}"
  if [[ ! -f "$conv" ]]; then
    mkdir -p "$(dirname "$conv")"
    (cd "$MT_DIR" && "$BENCH_PY" convert_sharegpt_to_openai.py "$SHAREGPT_PATH" "$conv" \
       --max-items 2000 --min-turns 4 --model "$TOKENIZER")
  fi
  echo "=== [$LABEL] $name ==="
  (cd "$MT_DIR" && "$BENCH_PY" benchmark_serving_multi_turn.py \
     --input-file "$conv" --model "$TOKENIZER" --served-model-name "$MODEL" --url "$BASE" \
     --num-clients "$c" --max-active-conversations "$(( c * 4 ))" \
     --max-num-requests "$(( c * PROMPTS_MULT ))" --limit-max-tokens 1024 \
     --send-conversation-id --conversation-sampling random --seed 0 \
     --stats-json-output "$OUT/$name.multiturn.json") 2>&1 | tee "$OUT/$name.log" \
    || echo "!!! $name failed (see log)"
}

for sc in $SCENARIOS; do
  for c in ${CONCURRENCIES:-${CONCS[$sc]}}; do
    for ((r = 1; r <= REPEATS; r++)); do
      name="${sc}_c${c}"; (( REPEATS > 1 )) && name+="_r$r"
      scrape "$OUT/$name.before.prom"
      if [[ "$sc" == multiturn ]]; then run_multiturn "$c" "$name"; else run_serve_bench "$sc" "$c" "$name"; fi
      scrape "$OUT/$name.after.prom"
    done
  done
done

"$BENCH_PY" "$ROOT/bench/summarize.py" "$OUT"
