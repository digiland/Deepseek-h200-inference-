#!/bin/bash
# Per-user-speed benchmark, v2: every rung uses its own prompt seed (V41_SEED_BASE + conc)
# so prompts are never already in the prefix cache, and records the server-side
# prefix-cache hit rate per rung. v1 used --seed 0 everywhere: 56-97% cache hits.
#
# usage: bench_latency2_v41.sh <tag> <port> <random|sharegpt> [conc ...]
#
# Per rung it records output tok/s, TTFT, TPOT p50/p95, speed per user
# (1000 / TPOT p50), mean output length, and, when the server runs speculative
# decoding, the mean acceptance length (tokens emitted per decode step per user)
# read from the server's /metrics counters before and after the rung.
#
# random   : 1024 random token ids in, 256 out, --ignore-eos. Same protocol as
#            rounds 1 and 2, so rows are comparable with results.csv.
# sharegpt : real ShareGPT conversations, output capped at 256, EOS honoured,
#            sampling at temperature $V41_TEMP. Draft acceptance on random ids is
#            not representative, so every speculative-decoding row uses this.
set -euo pipefail
S=/data/nvme3/shona-cpt; D=/data/nvme3/lab; T=$D/teachers
M=$T/deepseek-v4.1-flash-1017
SERVED=deepseek-ai/DeepSeek-V4.1-Flash-1017
SHAREGPT=${SHAREGPT:-$T/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json}
TEMP=${V41_TEMP:-0.6}
TAG=${1:?usage: bench_latency2_v41.sh <tag> <port> <random|sharegpt> [conc ...]}
PORT=${2:?}; DS=${3:?}; shift 3 || true
CONC=("$@"); [ ${#CONC[@]} -eq 0 ] && CONC=(8 25 50 100)
OUT=/data/nvme3/dfly2/results/bench/$TAG; mkdir -p "$OUT"
CSV=/data/nvme3/dfly2/results/bench/results.csv
[ -f "$CSV" ] || echo "tag,dataset,conc,num_prompts,output_tok_s,ttft_p50_ms,tpot_p50_ms,tpot_p95_ms,per_user_tok_s,mean_out_len,accept_len,prefix_hit_pct" > "$CSV"
SEED_BASE=${V41_SEED_BASE:-1000}
. "$S/enroot/env.sh"

scrape() {  # print "drafts accepted" from /metrics, or "na na"
  curl -sf "http://127.0.0.1:${PORT}/metrics" 2>/dev/null | python3 -c '
import re, sys
d = a = None
q = h = 0.0
for line in sys.stdin:
    if line.startswith("#"):
        continue
    m = re.match(r"vllm:spec_decode_num_drafts(?:_total)?(?:\{[^}]*\})?\s+([\d.e+]+)", line)
    if m: d = (d or 0) + float(m.group(1))
    m = re.match(r"vllm:spec_decode_num_accepted_tokens(?:_total)?(?:\{[^}]*\})?\s+([\d.e+]+)", line)
    if m: a = (a or 0) + float(m.group(1))
    m = re.match(r"vllm:prefix_cache_queries(?:_total)?(?:\{[^}]*\})?\s+([\d.e+]+)", line)
    if m: q = q + float(m.group(1))
    m = re.match(r"vllm:prefix_cache_hits(?:_total)?(?:\{[^}]*\})?\s+([\d.e+]+)", line)
    if m: h = h + float(m.group(1))
print(d if d is not None else "na", a if a is not None else "na", q, h)
' || echo "na na 0 0"
}

for c in "${CONC[@]}"; do
  n=$(( c * ${V41_NMULT:-8} )); [ "$n" -lt 400 ] && n=400
  if [ "$DS" = random ]; then
    DARGS=(--dataset-name random --random-input-len 1024 --random-output-len 256 --ignore-eos)
  else
    DARGS=(--dataset-name sharegpt --dataset-path "$SHAREGPT" --sharegpt-output-len 256
           --temperature "$TEMP" --top-p 0.95)
  fi
  read -r d0 a0 q0 h0 < <(scrape)
  echo "[$(date -Is)] bench $TAG $DS c=$c n=$n"
  enroot start -m /data/nvme3:/data/nvme3 -e HF_HOME=$S/hf_cache vllm \
    vllm bench serve --backend openai-chat \
      --endpoint /v1/chat/completions \
      --host 127.0.0.1 --port "$PORT" \
      --model "$SERVED" --tokenizer "$M" \
      "${DARGS[@]}" \
      --num-prompts "$n" --max-concurrency "$c" --seed $(( SEED_BASE + c )) \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
      ${BENCH_EXTRA:+--extra-body "{\"chat_template_kwargs\": $BENCH_EXTRA}"} \
      --save-result --result-dir "$OUT" --result-filename "${DS}_c${c}.json" \
      > "$OUT/${DS}_c${c}.log" 2>&1 || { echo "WARN: bench failed for $TAG $DS c=$c"; continue; }
  read -r d1 a1 q1 h1 < <(scrape)

  python3 - "$OUT/${DS}_c${c}.json" "$TAG" "$DS" "$c" "$CSV" "$d0" "$a0" "$d1" "$a1" "$q0" "$h0" "$q1" "$h1" <<'EOF' || \
    echo "WARN: could not parse results for $TAG $DS c=$c"
import json, sys
p, tag, ds, conc, csv, d0, a0, d1, a1, q0, h0, q1, h1 = sys.argv[1:14]
d = json.load(open(p))
tpot = float(d.get("median_tpot_ms") or 0)
done = d.get("completed") or 0
mean_out = (d.get("total_output_tokens") or 0) / done if done else 0
acc = ""
try:
    dd, da = float(d1) - float(d0), float(a1) - float(a0)
    if dd > 0:
        acc = f"{1 + da / dd:.2f}"      # tokens emitted per step per user
except ValueError:
    pass
row = [tag, ds, conc, str(d.get("num_prompts", 0)),
       f"{d.get('output_throughput', 0):.1f}",
       f"{d.get('median_ttft_ms', 0):.0f}",
       f"{tpot:.1f}",
       f"{d.get('p95_tpot_ms', 0):.1f}",
       f"{1000 / tpot:.1f}" if tpot else "",
       f"{mean_out:.0f}", acc]
dq = float(q1) - float(q0)
row.append(f"{100 * (float(h1) - float(h0)) / dq:.1f}" if dq > 0 else "")
print("  %-10s %-8s conc=%-4s out=%8.1f tok/s  ttft50=%6s ms  tpot50=%5s ms  per_user=%5s tok/s  accept=%s  hit=%s%%"
      % (tag, ds, conc, float(row[4]), row[5], row[6], row[8], acc or "-", row[11] or "-"))
open(csv, "a").write(",".join(row) + "\n")
EOF
done
