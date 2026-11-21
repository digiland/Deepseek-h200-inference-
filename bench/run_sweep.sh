#!/usr/bin/env bash
# Engine-neutral throughput sweep against any OpenAI-compatible server
# (vLLM, SGLang, ...). Uses `vllm bench serve` as the one load generator,
# so every engine is measured by exactly the same client.
#
# Usage:
#   bench/run_sweep.sh <label> [scenario ...]
#
#   label     tag such as "vllm-baseline" or "vllm-max"; results go to
#             results/<label>/<scenario>_c<concurrency>.json
#   scenario  any of the names below (default: decode chat sharegpt prefix)
#
#   smoke     512 in / 128 out, c=8                       (is it alive?)
#   decode    1k in / 1k out, random tokens               (pure decode throughput)
#   chat      4k in / 1k out, random tokens
#   long      32k in / 1k out, random tokens               (long context)
#   prefill   8k in / 64 out                               (prefill throughput)
#   sharegpt  real conversations, natural lengths <=1k out (realistic DSpark acceptance)
#             needs SHAREGPT_PATH=/path/ShareGPT_V3_unfiltered_cleaned_split.json
#   prefix    16 shared 4k prefixes + 512 unique suffix, 512 out  (prefix-cache hit rate)
#   mm        chat endpoint, 1 random image + 512 tokens text, 256 out (multimodal path)
#
# Env overrides:
#   HOST/PORT          server address (default 127.0.0.1:8000)
#   MODEL              served model name (default: first entry of /v1/models)
#   TOKENIZER          tokenizer path/id (default: $MODEL_PATH, else MODEL)
#   CONCURRENCIES      override concurrency list, e.g. "64 256"
#   NUM_PROMPTS_MULT   prompts per concurrency level (default 4 -> 4*c prompts)
#   BENCH_PY           python that has vllm installed (default: python3)
set -euo pipefail

LABEL=${1:?usage: run_sweep.sh <label> [scenario ...]}
shift || true
SCENARIOS=${*:-"decode chat sharegpt prefix"}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
BENCH_PY=${BENCH_PY:-python3}
NUM_PROMPTS_MULT=${NUM_PROMPTS_MULT:-4}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$ROOT/results/$LABEL}
mkdir -p "$OUT"

if [[ -z "${MODEL:-}" ]]; then
  MODEL=$(curl -s "http://$HOST:$PORT/v1/models" | "$BENCH_PY" -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
fi
TOKENIZER=${TOKENIZER:-${MODEL_PATH:-$MODEL}}

{
  echo "date: $(date -Is)"
  echo "label: $LABEL"
  echo "model: $MODEL"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,clocks.max.sm --format=csv,noheader 2>/dev/null || true
} > "$OUT/env.txt"
[[ -f "$ROOT/results/.last_server_cmd" ]] && cp "$ROOT/results/.last_server_cmd" "$OUT/server_cmd.txt" || true

# scenario -> default concurrencies
declare -A CONCS=(
  [smoke]="8" [decode]="64 128 256 512 1024" [chat]="64 128 256 512" [long]="16 32 64 128"
  [prefill]="32 64 128" [sharegpt]="64 128 256 512 1024" [prefix]="64 256 512" [mm]="32 64 128"
)

dataset_args() {  # prints the dataset-specific flags for a scenario
  case "$1" in
    smoke)   echo --dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos ;;
    decode)  echo --dataset-name random --random-input-len 1024 --random-output-len 1024 --ignore-eos ;;
    chat)    echo --dataset-name random --random-input-len 4096 --random-output-len 1024 --ignore-eos ;;
    long)    echo --dataset-name random --random-input-len 32768 --random-output-len 1024 --ignore-eos ;;
    prefill) echo --dataset-name random --random-input-len 8192 --random-output-len 64 --ignore-eos ;;
    sharegpt)
      : "${SHAREGPT_PATH:?set SHAREGPT_PATH to ShareGPT_V3_unfiltered_cleaned_split.json}"
      echo --dataset-name sharegpt --dataset-path "$SHAREGPT_PATH" --sharegpt-output-len 1024 ;;
    prefix)  echo --dataset-name prefix_repetition --prefix-repetition-prefix-len 4096 \
                  --prefix-repetition-suffix-len 512 --prefix-repetition-num-prefixes 16 \
                  --prefix-repetition-output-len 512 --ignore-eos ;;
    mm)      echo --backend openai-chat --endpoint /v1/chat/completions \
                  --dataset-name random-mm --random-input-len 512 --random-output-len 256 \
                  --random-mm-base-items-per-request 1 --ignore-eos ;;
    *) echo "unknown scenario $1" >&2; return 1 ;;
  esac
}

for sc in $SCENARIOS; do
  dargs=$(dataset_args "$sc")
  backend_args="--backend openai --endpoint /v1/completions"
  [[ "$dargs" == *"--backend"* ]] && backend_args=""
  for c in ${CONCURRENCIES:-${CONCS[$sc]}}; do
    n=$(( c * NUM_PROMPTS_MULT ))
    (( n < 32 )) && n=32
    echo "=== [$LABEL] $sc concurrency=$c prompts=$n ==="
    # shellcheck disable=SC2086
    "$BENCH_PY" -m vllm.entrypoints.cli.main bench serve \
      $backend_args \
      --base-url "http://$HOST:$PORT" \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --trust-remote-code \
      $dargs \
      --num-prompts "$n" \
      --max-concurrency "$c" \
      --request-rate inf \
      --num-warmups "$(( c < 16 ? c : 16 ))" \
      --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 50,90,99 \
      --save-result \
      --result-dir "$OUT" \
      --result-filename "${sc}_c${c}.json" \
      2>&1 | tee "$OUT/${sc}_c${c}.log" || echo "!!! $sc c=$c failed (see log)"
  done
done

# Server-side counters: prefix-cache hit rate and spec-decode acceptance (vLLM metric names).
curl -s "http://$HOST:$PORT/metrics" > "$OUT/server_metrics.prom" 2>/dev/null || true
"$BENCH_PY" "$ROOT/bench/summarize.py" "$OUT"
