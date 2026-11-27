#!/bin/bash
# Build the vLLM-servable DFly2 drafter dir: released weights (AngelSlim/DS-V4.1-Flash-DFly2-B6)
# + this config.json (architecture DSparkV41DraftModel and dspark_* fields for vLLM 0.32.0's
# DSpark loader) + tokenizer, chat template and encoding from the target model.
# usage: make_served_dir.sh <release dir> <target model dir> <out dir>
set -euo pipefail
REL=$1; TGT=$2; OUT=$3; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
for f in "$REL"/model-*.safetensors "$REL"/model.safetensors.index.json; do ln -sf "$f" "$OUT/"; done
for f in chat_template.jinja encoding inference LICENSE README.md tokenizer_config.json tokenizer.json; do ln -sfn "$TGT/$f" "$OUT/$f"; done
cp "$HERE/config.json" "$OUT/config.json"
echo "served drafter dir: $OUT"
