#!/usr/bin/env bash
# Engine-neutral throughput sweep against any OpenAI-compatible server
# (vLLM, SGLang, ...). Uses `vllm bench serve` as the single load generator
# so that every engine is measured by the exact same client.
#
# Usage:
#   bench/run_sweep.sh <label> [scenario ...]
#
#   label     free-form tag, e.g. "vllm-baseline", "sglang-opt"; results go
#             to results/<label>/<scenario>_c<concurrency>.json
#   scenario  one or more of: decode chat long prefill smoke (default: decode chat long)
#
# Env overrides:
#   HOST/PORT          server address               (default 127.0.0.1:8000)
#   MODEL              served model name            (default: first from /v1/models)
#   TOKENIZER          tokenizer path/id            (default: $MODEL_PATH or MODEL)
#   CONCURRENCIES      override concurrency list, e.g. "64 256"
#   NUM_PROMPTS_MULT   prompts per concurrency level (default 4  -> 4*c prompts)
#   BENCH_PY           python that has vllm installed (default: python)
set -euo pipefail

LABEL=${1:?usage: run_sweep.sh <label> [scenario ...]}
shift || true
SCENARIOS=${*:-"decode chat long"}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
BENCH_PY=${BENCH_PY:-python}
NUM_PROMPTS_MULT=${NUM_PROMPTS_MULT:-4}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$ROOT/results/$LABEL}
mkdir -p "$OUT"

if [[ -z "${MODEL:-}" ]]; then
  MODEL=$(curl -s "http://$HOST:$PORT/v1/models" | "$BENCH_PY" -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
fi
TOKENIZER=${TOKENIZER:-${MODEL_PATH:-$MODEL}}

# scenario -> "input_len output_len default_concurrencies"
declare -A SPEC=(
  [smoke]="512 128 8"
  [decode]="1024 1024 64 128 256 512"
  [chat]="4096 1024 64 128 256"
  [long]="32768 1024 16 32 64"
  [prefill]="8192 64 32 64 128"
)

record_env() {
  {
    echo "date: $(date -Is)"
    echo "label: $LABEL"
    echo "model: $MODEL"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,clocks.max.sm --format=csv,noheader 2>/dev/null || true
  } > "$OUT/env.txt"
  # Snapshot the server-side config if the launcher left one behind.
  [[ -f "$ROOT/results/.last_server_cmd" ]] && cp "$ROOT/results/.last_server_cmd" "$OUT/server_cmd.txt" || true
}
record_env

for sc in $SCENARIOS; do
  read -r -a parts <<< "${SPEC[$sc]:?unknown scenario $sc}"
  in_len=${parts[0]}; out_len=${parts[1]}
  concs=${CONCURRENCIES:-"${parts[*]:2}"}
  for c in $concs; do
    n=$(( c * NUM_PROMPTS_MULT ))
    (( n < 32 )) && n=32
    echo "=== [$LABEL] $sc in=$in_len out=$out_len concurrency=$c prompts=$n ==="
    "$BENCH_PY" -m vllm.entrypoints.cli.main bench serve \
      --backend openai \
      --base-url "http://$HOST:$PORT" \
      --endpoint /v1/completions \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --trust-remote-code \
      --dataset-name random \
      --random-input-len "$in_len" \
      --random-output-len "$out_len" \
      --random-range-ratio 0.0 \
      --ignore-eos \
      --num-prompts "$n" \
      --max-concurrency "$c" \
      --request-rate inf \
      --num-warmups "$(( c < 16 ? c : 16 ))" \
      --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 50,90,99 \
      --save-result \
      --result-dir "$OUT" \
      --result-filename "${sc}_c${c}.json" \
      2>&1 | tee "$OUT/${sc}_c${c}.log"
  done
done

"$BENCH_PY" "$ROOT/bench/summarize.py" "$OUT"
