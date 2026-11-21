# DeepSeek-V4.1-Flash-1017 on 4×H200: throughput work

Goal: **≥ 8,000 output tok/s** from one 4×H200 node for many concurrent users, with prefix caching and multimodal input kept. That should be 2× or more over the stock vLLM recipe.

- **Plan and reasoning:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Hardware notes:** [`docs/H200.md`](docs/H200.md)

## Layout

| Path | What |
|---|---|
| `third_party/{vllm,sglang,flashinfer,DeepInfer}` | Upstream sources pinned as submodules (read-only reference; serving uses the Docker images) |
| `h200opt/` | vLLM plugin: native FP8 DeepGEMM path for V4.1's MXFP8 dense layers on SM90, plus opt-in SM90 threshold knobs. CPU tests in `h200opt/tests`, GPU tests in `h200opt/tests/gpu` |
| `serve/vllm/launch.sh` | vLLM profiles: baselines (`baseline`, `baseline-spec`, `tp4-tuned`, `tp4-tuned-spec`) and the ladder `dep4` → `dep4-mega` → `dep4-mega-spec` → `max` / `max-nospec` |
| `serve/router/affinity_router.py` | Prefix-affinity router for DP4: pins each conversation to one DP rank so its prefix cache is reused |
| `serve/sglang/launch.sh` | SGLang profiles: `baseline`, `dpa4`, `dpa4-nospec`, `dpa4-hicache` |
| `docker/` | `Dockerfile.vllm` (official vLLM image + plugin), `Dockerfile.sglang`, `run.sh` (build/up/logs/down) |
| `bench/run_sweep.sh` | Load generator (`vllm bench serve` and vLLM's multi-turn benchmark, the same client for every engine). Headline: sharegpt, multiturn, mm; synthetic: decode, chat, long, prefill, prefix |
| `bench/summarize.py` | Per-run tables, A-vs-B speedup tables, per-run prefix-cache hit rate and DSpark acceptance |
| `bench/quality_gate.py` | Greedy-logprob agreement vs a reference profile + GSM8K accuracy; fails the profile on a regression |
| `scripts/ladder.sh` | Runs profiles back to back (start → wait → bench → stop) and writes comparisons |
| `scripts/check_node.sh` | Node inventory: GPUs, NVLink topology, driver, host RAM, Python env |
| `tools/reblock_report.py` | CPU-only check of the FP8 weight re-blocking error on the real checkpoint (weights only; output quality is `quality_gate.py`'s job) |

## Quickstart (on the H200 node)

```bash
git clone --recurse-submodules <this repo> && cd Deepseek-h200-inference-   # submodules are optional for serving
scripts/check_node.sh > results/node_info.txt 2>&1

# 1. Weights (~510 GB) and a realistic prompt set
huggingface-cli download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir /data/DeepSeek-V4.1-Flash-1017
huggingface-cli download anon8231489123/ShareGPT_Vicuna_unfiltered ShareGPT_V3_unfiltered_cleaned_split.json \
    --repo-type dataset --local-dir /data/sharegpt
export MODEL_DIR=/data/DeepSeek-V4.1-Flash-1017
export SHAREGPT_PATH=/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json

# 2. Build the image (official vLLM + h200opt plugin)
docker/run.sh build vllm

# 3. Pre-flight checks (minutes; both run inside the image, which has torch/safetensors/pytest)
docker run --rm --entrypoint python3 -v $MODEL_DIR:/model:ro -v $PWD/results:/opt/h200/results \
    h200/vllm:latest tools/reblock_report.py /model --out results/reblock.csv        # CPU only
docker run --rm --gpus '"device=0"' --entrypoint python3 -v $PWD/h200opt:/src h200/vllm:latest \
    -m pytest -q /src/tests/gpu -s                                                   # 1 GPU

# 4. The ladder (~1 h per profile incl. weight load; quality gate + benchmarks per profile)
scripts/ladder.sh                          # LADDER=core: 3 baselines -> dep4-mega-spec -> max -> max-nospec
LADDER=extended scripts/ladder.sh          # full per-lever attribution + SGLang cross-check
scripts/ladder.sh vllm:tp4-tuned-spec 'vllm:max-nospec;NODE_SEQS=4096'   # ad hoc, ';KEY=VAL' overrides
```

Serve a single profile by hand:

```bash
MODEL_DIR=/data/DeepSeek-V4.1-Flash-1017 docker/run.sh up vllm max     # OpenAI API on :8000
docker/run.sh logs vllm
bench/run_sweep.sh my-test sharegpt mm                                 # needs vllm on the host,
                                                                       # or run it inside the image
```

Useful knobs (env vars, passed through by `docker/run.sh`):
- `NODE_SEQS`: concurrent sequences for the whole node (default 1024; DP profiles split it over 4 ranks)
- `MAX_MODEL_LEN` (default 131072)
- `SPEC_K` (default 5)
- `SPEC_SCHEDULE`: drafts per batch-size range, e.g. `'[[1,64,5],[65,192,3],[193,1024,1]]'`
- `ADAPTIVE=0`: disable adaptive verification
- `INDEXER_BF16=1`, `EPLB=1`, `TEXT_ONLY=1`
- `H200OPT_MXFP8_EXCLUDE`: regex of dense layers kept BF16 under `max` (default `indexer|compressor`)
- `EXTRA_ARGS`
- `PASS_ENV="H200OPT_SM90_MHC_MAX_TOKENS ..."`: forward plugin env vars

## What to send back

Send everything under `results/`: `node_info.txt`, `reblock.csv`, the GPU test output, each `results/<label>/summary.md`, `results/compare_*.md`, `results/quality-*.txt`, and `results/server-*.log` for any profile that failed to start.

The first version of this plan was reviewed independently; what it caught and what changed is in `docs/ARCHITECTURE.md` §9.
