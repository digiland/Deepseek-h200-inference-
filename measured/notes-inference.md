# Notes: DeepSeek V4.1 Flash inference on 8 x H200

Started 23 Nov 2026. Host `cpt-gpu170` (8 x NVIDIA H200 141 GB, 192 cores, 2 TiB RAM).
Goal: serve `deepseek-ai/DeepSeek-V4.1-Flash-1017` with the vLLM H200 recipe, then tune the node for maximum output tokens per second. Reference target from the digiland H200 plan: 8,000 output tok/s on four H200 cards.

All numbers in this file were read from the result files under `/data/nvme3/lab/teachers/bench/v41/`.

---

## 1. What was set up

| Item | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4.1-Flash-1017`, 476 GiB, 48 safetensors shards |
| Local path | `/data/nvme3/lab/teachers/deepseek-v4.1-flash-1017` |
| Downloaded | 23 Nov 2026, `hf download`, 48/48 shards, 1.2 TB free left |
| Engine | vLLM 0.32.0, inside enroot container `vllm` (image `vllm-0.32.0.sqsh`) |
| vLLM is not on the host | it exists only in the enroot image; the host `python3` has no vllm |
| Rust frontend | `/usr/local/bin/vllm-rs` present, so `VLLM_USE_RUST_FRONTEND=1` is usable |
| Architecture support | `DeepseekV41ForCausalLM` registered; v41 reasoning parser and v41 tool parser registered |
| Host caches | `VLLM_CACHE_ROOT=/data/nvme3/shona-cpt/cache/vllm`, Triton and XDG caches beside it |

### The recipe used

Source: `https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash?hardware=h200&strategy=single_node_tp`
(hardware `h200`, strategy `single_node_tp`, variant FP8 1017.)

Flags applied verbatim:

```
--tensor-parallel-size 4
--engram-config '{"cpu_offload":true}'
--tool-call-parser deepseek_v41 --enable-auto-tool-choice
--reasoning-parser deepseek_v41
--mm-encoder-tp-mode data
VLLM_ENGINE_READY_TIMEOUT_S=3600
VLLM_USE_RUST_FRONTEND=1
```

One deliberate deviation, `--tokenizer-mode deepseek_v41` is omitted. Reason: in vLLM 0.32.0 that value is not in the flag's argparse choices (`auto, cohere, deepseek_v32, deepseek_v4, hf, inkling, kimi_k3, mistral`), but `ModelConfig` assigns `tokenizer_mode = "deepseek_v41"` automatically when the architecture is `DeepseekV41ForCausalLM` (`vllm/config/model.py:751`). Passing the flag explicitly would fail at startup; `auto` gives the same result.

Recipe's H200 budget: 81.2 GiB of resident weights and 38.5 GiB of KV per GPU at TP4, about 20M tokens of KV.

---

## 2. Scripts written

All live in `/data/nvme3/lab/teachers/`.

| File | Purpose |
|---|---|
| `serve-deepseek-v4.1-flash.sh` | start the server with the recipe flags. Env overrides: `V41_PORT`, `V41_TP`, `V41_GPUS`, extra args passed through |
| `bench_v41.sh` | throughput bench, `vllm bench serve`, openai-chat backend, random 1024-in/256-out, writes JSON plus a CSV row |
| `tune_v41.sh` | round 1 sweep: start, wait for health, bench, stop, next variant |
| `bench_dual.sh` | two TP4 replicas on GPUs 0-3 and 4-7, aggregate tok/s |
| `run_tp8.sh` | one replica across all 8 GPUs, `V41_TP=8` |
| `run_round2_v41.sh` | round 2: `ctx128k`, `budget`, `budget-spec`, `budget2k` |
| `chain_all_v41.sh` | sequential tail: dual, tp8, round 2, finalise |
| `monitor_v41.sh` | watchdog, checks every 10 minutes, restarts a dead or stalled chain, exits on completion |
| `finalise_v41.sh` | copy artifacts to `/data/nvme3/inference`, rebuild the report |
| `summarise_v41.py` | print the results table and the peak |
| `make_report_v41.py` | build `index.html` from the CSVs using `lab/report/theme.py` |

### How the bench works

```
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions
  --model deepseek-ai/DeepSeek-V4.1-Flash-1017 --tokenizer <local model dir>
  --dataset-name random --random-input-len 1024 --random-output-len 256
  --num-prompts 8 x concurrency (minimum 400) --max-concurrency <c>
  --ignore-eos --seed 0 --percentile-metrics ttft,tpot,itl,e2el
  --metric-percentiles 50,95,99 --save-result
```

Reported per cell: output tok/s, requests/s, TTFT p50, TPOT p50.

---

## 3. Bugs found and fixed

| Problem | Symptom | Fix |
|---|---|---|
| Bench URL | `ValueError: OpenAI Chat Completions API URL must end with one of: {'chat/completions'}` because the default endpoint is `/v1/completions` | added `--endpoint /v1/chat/completions` |
| Bench tokenizer mode | bench given `--tokenizer-mode deepseek_v41`, which this build does not list | dropped it; `auto` resolves the tokenizer from the model directory |
| Server tokenizer mode | explicit `deepseek_v41` rejected by argparse | rely on `auto`, documented above |
| Triton expert backend | `moe_backend='triton' is not supported for MXFP4 MoE.` (`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:384`), engine core init failed | variant recorded as failed; MXFP4 on SM90 has no Triton path, `auto` must choose another kernel |
| Editing a running script | I edited `bench_dual.sh` while bash was executing it. Bash reads scripts by byte offset, so the edit corrupted the instruction stream (`one: command not found`) and it hung 76 minutes holding both replicas | killed the chain, confirmed GPUs back to 0 MiB, removed the two orphaned rows, restarted the tail as one sequential script. No mid-run edits since |
| `pkill -f` self-match | two of my monitoring shells were killed by their own `pkill -f` pattern (the pattern text appears in the shell's command line) | list PIDs first, kill by PID; use bracket patterns such as `chain_all_v41[.]sh` |
| `bench_dual` hang | a bare `wait` waited for the two long-lived server processes as well as the two bench jobs, so the dual stage blocked at the first concurrency level for 39 minutes | capture both bench PIDs and use `wait "$p1" "$p2"` |
| Dry-run row | a client test against a dead port wrote a zero-throughput row | removed `bench/v41/dryrun` and the row |

---

## 4. Round 1 results

Output tokens per second, single model on 4 GPUs. n/r means that rung was not run.

| Variant | What changed | Conc 1 | Conc 8 | Conc 64 | Conc 256 |
|---|---|---:|---:|---:|---:|
| `recipe` | nothing, recipe defaults | n/r | 690.3 | 2,470.0 | 3,683.8 |
| `kernel` | `enable_flashinfer_autotune` | n/r | 715.4 | 2,480.4 | 3,992.3 |
| `seqs256` | `--max-num-seqs 256` | n/r | 705.5 | 2,511.8 | 3,969.8 |
| `batchtok16k` | `--max-num-batched-tokens 16384` | n/r | 712.6 | 2,562.6 | **4,100.5** |
| `gpu95` | `--gpu-memory-utilization 0.95` | n/r | 714.6 | 2,524.3 | 3,988.1 |
| `dspark` | DSpark, 5 drafts, probabilistic | 201.9 | **861.8** | 1,978.5 | n/r |

Latency at the peak row: TTFT p50 1,518 ms, TPOT p50 59.8 ms, 16.02 requests/s.

### What round 1 established

- FlashInfer autotune is a real gain: +8.4% at conc 256 (3,992 against 3,684) with lower TTFT (905 ms against 1,120 ms).
- `batchtok16k` is the throughput maximum at 4,100 tok/s, paid for in TTFT (1,518 ms against 905 ms).
- `seqs256` adds nothing over `kernel`. The scheduler was not starved of sequences.
- `gpu95` does not help. The node is compute bound, not KV bound.
- DSpark helps small batches and hurts large ones: +25% at conc 8 (862 against 690), −20% at conc 64 (1,979 against 2,470). This matches the digiland roofline prediction.
- GPU memory sat at 136 to 139 GB of 141 GB during all runs, which is consistent with a large fixed buffer rather than a KV limit.

---

## 5. What round 2 tests, and why

Source: `https://github.com/digiland/Deepseek-h200-inference-`, `docs/ARCHITECTURE.md` and `serve/vllm/launch.sh`. That plan reports no H200 measurements yet; its 8,000 tok/s figure is a roofline estimate.

| Variant | Change | Reason |
|---|---|---|
| `ctx128k` | `--max-model-len 131072` | the indexer logits buffer is `batched_tokens x max_len x 4 bytes`. At 1,048,576 tokens that is about 34 GB per GPU; at 131,072 about 4 GB. Freeing it is what allows a large batch |
| `budget` | 128K, `--max-num-seqs 1024`, `--max-cudagraph-capture-size 2048`, `--gpu-memory-utilization 0.92` | batch budget and CUDA graphs sized to the decode step |
| `budget-spec` | budget plus DSpark with `rejection_sample_method:block`, adaptive verification, drafts `[[1,64,5],[65,192,3],[193,1024,1]]` | DSpark that gives back drafts as the batch grows |
| `budget2k` | 128K, `--max-num-seqs 2048`, benched to concurrency 2048 | the plan says 2,000 tok/s per GPU needs several hundred sequences per GPU |

Ladder for round 2: 64, 256, 512, 1024, and 256 to 2048 for `budget2k`.

### GPU layout comparison

| Run | Layout | Purpose |
|---|---|---|
| round 1 and 2 | one model, TP4, 4 GPUs | the recipe layout |
| `tp8` | one model, TP8, 8 GPUs | does one big replica beat one small one |
| `dual_tp4` | two models, TP4 each, 8 GPUs | does two replicas beat one big one |

---

## 6. Known limitations of these numbers

- The prompt set is `random`, which vLLM generates as consecutive token ids. The digiland plan warns this makes DSpark acceptance, Engram lookups and the indexer unrepresentative. These results are valid for ranking variants against each other, not as a forecast of real-text throughput. ShareGPT was not downloaded.
- One run per cell. No repeats, so differences under about 2% are noise.
- The latencies quoted are medians from a single run, not p99.
- The 8,000 tok/s target in the digiland plan is measured against its own tuned baselines, not against this `random` protocol.

---

## 7. Watchdog behaviour

`monitor_v41.sh` checks every 10 minutes and appends one line to `teachers/monitor-v41.log`:

| Condition | Action |
|---|---|
| chain process alive | log heartbeat, continue |
| chain dead before `/data/nvme3/inference/README.txt` exists | restart `chain_all_v41.sh`, maximum 2 restarts |
| no log growth for 30 minutes and total GPU utilisation under 5% | kill the API servers and the chain, restart, maximum 2 restarts |
| finalise marker present and chain exited | exit 0 |

The stall guard needs three consecutive idle checks because a 476 GiB load can sit at 0% GPU for several minutes.

---

## 8. File layout of the result folder

```
/data/nvme3/inference/
├── index.html         summary page, rebuilt from the final CSVs
├── notes-infrence.md  this file
├── results.csv        single-node variants by concurrency
├── results_dual.csv   two-replica comparison
├── README.txt         what ran and where the scripts are
├── raw/               per-run bench JSON and logs
└── logs/              server logs, master log, gpus.csv, disk.txt
```

## 9. How to reproduce

```bash
cd /data/nvme3/lab/teachers
bash serve-deepseek-v4.1-flash.sh                    # recipe server on 127.0.0.1:8105
bash bench_v41.sh mytag 8105 8 64 256                # bench a running server
bash tune_v41.sh all                                 # round 1 sweep
bash run_tp8.sh                                      # one model on 8 GPUs
bash bench_dual.sh                                   # two replicas on 8 GPUs
bash run_round2_v41.sh                               # round 2
bash finalise_v41.sh                                 # rebuild the folder and report
```

## 10. Timeline

| Time (UTC, 23 Nov 2026) | Event |
|---|---|
| 07:09 | machine and existing work inspected; 8 H200 idle |
| 07:15 | V4.1 download started, 511 GB |
| 07:25 | recipe, bench and tuning scripts written; vLLM 0.32.0 confirmed in enroot |
| 07:36 | download complete, 48 of 48 shards |
| 07:43 | smoke test passed: `17*19` returned `323`, engine `vllm-0.32.0-tp4-135ed764` |
| 07:48 | first sweep aborted, bench client endpoint bug found |
| 07:51 | sweep restarted with the endpoint fix |
| 09:04 | round 1 single-node sweep done, dual stage started |
| 10:21 | dual stage found hung after 76 minutes, caused by an edit to a running script |
| 10:22 | tail restarted as `chain_all_v41.sh` |
| 10:24 | watchdog started |
| 11:01 | dual stage hung again, this time because a bare `wait` blocked on the two server processes; killed the chain and both replicas |
| 11:06 | patched `bench_dual.sh` to wait only on the two bench jobs; chain and watchdog restarted |

Final results are appended below when the run completes.

---

## 11. Final results, speed per user (23 Nov 2026, 17:45 to 23:44 UTC)

Report: `report/DeepSeek V4.1 Inference TR-01 - Speed per user and total output on 8 x H200 (24 Nov 2026).pdf` (two pages, every number generated from `latency/`).

| Question | Result |
|---|---|
| Most users at >= 80 tok/s per user | 48 users at 81.7 tok/s each, 3,614 tok/s total: 1 copy on 8 GPUs, W4A8 (FP8 activations) + DSpark 3 drafts. 52 users: 78.2 tok/s |
| Highest total output | 18,788 tok/s at 2,048 users (9.6 tok/s each): 2 copies on 4 GPUs, W4A8, no drafts. 15,312 tok/s at 1,024 users (16.4 tok/s each) |
| Benchmark used | ShareGPT, output up to 256, fresh prompt seed per run (`bench_latency2_v41.sh`); the first latency benchmark reused `--seed 0` and hit the prefix cache 56 to 97% |
| What helped | DSpark 3 drafts (2.5 tokens per step per user); W4A8 via `VLLM_HUMMING_INPUT_QUANT_CONFIG` (+10 to 11% with DSpark, same answers on the 50-question probe) |
| What did not | Marlin / CUTLASS / grouped MoE, all-reduce backends, Engram on GPU, 1,024-token prompt chunks, DP8 + EP (allgather), Humming padding 128 |
| Could not start | DeepEP (no InfiniBand in the container), MegaMoE (SM100 only), FlashInfer sparse MLA (block size conflict with the V4.1 indexer), Humming padding 64, FP16 accumulation |
| Decode step, 8 GPUs, 200 users | 26.6 ms GPU busy (MoE GEMM 16.2, attention 4.5, all-reduce 1.8, mHC 1.7); weight reads alone 7.4 ms |

Copies of the CSVs, quality probes and trace analyses: `latency/`. Scripts: `/data/nvme3/lab/teachers/run_*_v41.sh`, `bench_latency2_v41.sh`, `quality_v41.py`, `analyse_trace_v41.py`, `patches/`.

## 12. Second round (24 Nov 2026, 00:00 to 02:37 UTC)

| Question | Result |
|---|---|
| Most users at >= 80 tok/s per user | 72 users at 84.8 tok/s each (5,594 tok/s): 2 copies on 4 GPUs, W4A8 + DSpark 3 drafts + Humming 2 CTAs per SM (decode buckets) + new prompts every 4th step. 80 users: 79.7 tok/s (inside noise of 80). Without the two patches: 64 users at 84.2 |
| 2 copies of 4 GPUs vs 1 copy of 8 | +17 to +29% tok/s per user at 40 to 200 users with W4A8 + DSpark |
| Patches (teachers/patches) | humming_tune/fused_humming_moe.py: tile overrides by env (2 CTAs/SM +5% at 48 users; 6 or 8 stages, stream-K: no change; tile 8x128x128: -13%; 8x256x256: did not start). prefill_interval/core.py: new prompts every N-th step (+1 to +7%, TTFT up to 2x). Humming pad 128: memory only, no speed |
| Trace, 48 users, DSpark | 26% of steps also process new prompts: 44.3 ms GPU busy vs 26.5 ms decode-only, 8.7 ms of it outside CUDA graphs |
| Operational | starting two copies at once stalled 21 min (concurrent kernel JIT); start them one after the other |

Report updated in place: `report/DeepSeek V4.1 Inference TR-01 ... (24 Nov 2026).pdf`.

## 13. Third round (24 Nov 2026, 02:50 to 03:36 UTC)

| Question | Result |
|---|---|
| Most users at >= 80 tok/s per user | **80 users at 82.0 tok/s** (5,833 tok/s): 2 copies on 4 GPUs, W4A8 + DSpark 3 drafts + 2 CTAs per SM + new prompts every 8th step. 88 users: 78.7. TTFT about 238 ms |
| Same with prompts every 4th step | 76 users at 81.8; 80 users at 79.8 (repeat of 79.7) |
| Dense MXFP8 layers with FP8 activations (patches/dense_w8a8) | starts and keeps quality (49/50) but no speed change (36 users 84.5 vs 84.9; 40 users 79.8 vs 79.8) |
| Fused mHC post + pre-norm kernel above its 32-token cap (patches/mhc_cap, cap 256) | 3 to 4% slower per user at 72 to 88 users (tuned for GB300); quality 49/50. Rejected |
| MoE backend triton / triton_unfused (exp 14) | triton: does not start (Triton MXFP4 MoE kernel lacks SiLU). triton_unfused: 39.7 ms at 100 users, 44.1 ms at 200 (Humming 27.0 / 34.1): 29 to 47% slower. Every MoE backend that runs V4.1 on H200 is now measured; Humming is fastest |
| Practical HBM bandwidth (hbm_bw_v41.py) | 4.3 TB/s per H200 (copy and read, 4 to 16 GiB). At 80 users and 82.0 tok/s, weight reads would take 6.4 ms per output token of the measured 12.2 ms (52%) |
| Repeat of the best setup, fresh prompts (exp 15) | 80 users: 81.6 tok/s (first run 82.0); 84 users: 79.7. Maximum at >= 80 tok/s per user: 80 users, confirmed twice |

## 14. Final sustained benchmark (24 Nov 2026, 07:58 to 09:09 UTC)

Real ShareGPT prompts, fresh seed per run, 16 x users requests per run (`run_final_v41.sh`, tags F1 to F4). Speed per user = 1,000 / TPOT p50.

| Users | 2 x 4 GPUs: tok/s per user | 2 x 4 GPUs: total tok/s | 1 x 8 GPUs: tok/s per user | 1 x 8 GPUs: total tok/s |
|---:|---:|---:|---:|---:|
| 40 (DSpark) | 109.0 | 3,985 | 94.5 | 3,412 |
| 80 (DSpark) | 81.7 | 5,946 | 67.4 | 4,896 |
| 160 (DSpark) | 56.4 | 8,242 | 44.2 | 6,436 |
| 200 (DSpark) | 50.5 | 9,225 | 39.1 | 7,106 |
| 768 (no drafts) | n/r | n/r | 13.1 | 9,274 |
| 1,024 (no drafts) | 15.7 | 15,236 | 9.0 | 8,632 |
| 2,048 (no drafts) | 8.8 | 17,367 | n/r | n/r |

Quality probe: 48 to 49 of 50 for every setup. 50 tok/s per user with a warm prefix cache (about 80% of prompt tokens cached, exp 16): 220 users. Custom CUDA expert GEMM (kernels/cuda_moe, TMA v5): correct, but 1.9x (down) and 2.5x (gate/up) slower than Humming, so not used.

## 15. GLM-5.3 (zai-org, 756 GB FP8) on 8 x H200 (26 Nov 2026)

Report: report/GLM 5.3 Inference TR-01 - Speed per user and total output on 8 x H200 (26 Nov 2026).pdf. Data: latency/glm53/.
Only fits as 1 copy x TP8. Best per-user: recipe + MTP 3 drafts (80.1 tok/s at 10 users, 26.9 at 80, 14.9 at 256). Max total 5,340 tok/s at 1,024 users (no drafts).
Expert parallel slower (-10% at 10 users), split KV cache (DCP8, needs --dcp-comm-backend ag_rs) gives 8.1x KV tokens but 20 to 34% slower on chat prompts, DP8+EP8 does not fit (1.14 GiB KV per GPU), DSpark draft model needs "kv_cache_dtype":"auto" and crashes at 80 users (vLLM FlashAttention scheduler_metadata bug).
Next: Flare SM90 port targets GLM-5.3-Flash (Flare's glm53 recipe is the Flash model, not this one).

## 16. FLARE (Hyperflare) SM90 port and DFly2 (26-27 Nov 2026)

**FLARE on H200, GLM-5.3-Flash (nvidia FP4-Local), 1 copy x TP4, sustained ShareGPT.** Port: /data/nvme3/flare-sm90 (branch sm90 in hyperflare-inference and hyperflare-recipes; b12x edits tracked in git inside site-packages; env.sh sets B12X_SM90_PTX=1). Quality 50/50 on every build.

| Build / flags | 10 users | 32 users | 64 users | 128 users | 256 users |
|---|---|---|---|---|---|
| Untuned 32-user core | 72.0 tok/s, TTFT 1,303 ms | - | - | - | - |
| Autotuned 64-user core | 100.3 tok/s, TTFT 1,237 ms | 874 tok/s total, TTFT 4,078 ms | - | - | - |
| + --scheduler-first-token-price 30 --scheduler-single-graph-prompts | 807 tok/s, TTFT 96 ms | 1,436 tok/s, TTFT 145 ms | - | - | - |
| 256-user core (same flags) | - | - | 1,745 tok/s | 1,986 tok/s | 1,648 tok/s |
| 256-user core, MTP off | - | - | - | 1,837 (64 ms step) | 1,878 (103 ms step) |
| vLLM 0.32.0 (nvidia NVFP4, TP4) | 704 / 1,071 MTP3, TTFT 89-145 ms | ~2,200 MTP3, TTFT 198 ms | 2,955 (80 users) | 4,273 (160 users) | 5,130 |

**Decode-step attribution at 256 users (one rank):** GPU forward 54 ms (routed experts 20.0, mHC 6.4, KDA layers, sparse MLA 3.5, projections 6.4, sampling 7.5 eager / ~0 in graph, all-reduce 1.4) vs measured 103 ms; ~49 ms between graphs (~0.2 ms per token, host side, not yet located). Expert kernels: b12x A16 21.3 ms/step vs Marlin 21.9, Humming NVFP4 19.2 (only 2-3 ms to gain). All-reduce: b12x already fastest. Conclusion: best case is about vLLM parity, not 2x; user stopped the FLARE work 27 Nov.

**Open bug:** cores built for >64 users intermittently fail with "rank N: tensor-parallel request epochs, consumption, or status differ" (root record error_code 32) under 256-user load; seen with both per-row and batched sampling.

**Kept changes (sm90 branch):** SM90 PTX rewriter (FP4/E4M3/UE8M0 conversions, FP8 block-scaled MMA emulation), A16 for all packed linears, block-FP8 via FlashInfer fp8_blockscale_gemm_sm90, Triton DSA indexer, gh100 platform, 256-lane scheduler (SlotMask), TTFT scheduler flags, batched row-independent sampling, 1,024-parameter core limit. Tools: tests/ (PTX, indexer, block-FP8, MMA emulation), moebench/ (kernel and component timings).

**DFly2 (Tencent AngelSpec) on vLLM, DeepSeek-V4.1-Flash, 1 copy x TP4, temp 0.6:** ported (/data/nvme3/dfly2), oracle match 97.9% top-1, quality 48-49/50. Per user at 20/40/80 users: DSpark-3 109.7 / 81.4 / 57.0 vs DFly2-3 98.1 / 72.6 / 51.6 (-11 to -12%); total at 80 users 3,658 vs 3,782. DSpark-3 stays the V4.1 choice.

**Storage cleanup 26 Nov:** 5.02 TB of optimizer state and intermediate checkpoints deleted from 25 finished jobs (list: lab/jobs/cleanup-2026-11-26.txt; LATEST/LOWEST_VAL and report-used adapters kept).

### 16.1 DFly2 vs DSpark full matrix, DeepSeek-V4.1-Flash-1017, 8 x H200 (27 Nov 2026)

Protocol: vLLM 0.32.0, F1 flags (W4A8 + Humming 2 CTAs/SM + prompts every 8th step), ShareGPT, T=0.6 top-p 0.95, output cap 256, 16 x users requests per rung, fresh seeds. Speeds are per node (2 x TP4 = both copies summed; TP8 = one copy). Summary: /data/nvme3/dfly2/results/matrix_summary.csv; rows: results/bench/results.csv; scripts: tools/run_matrix.sh, run_matrix_redo.sh, bench_any.sh, serve_v41_any.sh.

**Speed per user (tok/s) on 2 x TP4:**

| Users per node | DSpark-3, think on | DFly2-3, think on | DFly2-4, think on | DSpark-3, think off | DFly2-3, think off |
|---|---|---|---|---|---|
| 40 | **109.0** | 95.7 | 92.8 | 108.6 | 103.0 |
| 80 | 81.3 | 71.9 | 70.2 | **82.2** | 77.7 |
| 160 | 56.1 | 50.9 | 48.5 | **58.2** | 55.3 |
| 200 | 50.6 | 45.5 | 43.1 | **52.2** | 49.8 |
| Node total at 200 users | 9,184 | 8,378 | 7,931 | **9,413** | 9,005 |
| Accepted tokens per step | 2.48 | 2.31 | 2.44 | 2.55 | 2.53 |

**1 x TP8, thinking on (speed per user / node total):** DSpark-3 96.2 / 67.3 / 45.1 / 38.7 tok/s at 40 / 80 / 160 / 200 users (7,026 total at 200); DFly2-3 86.0 / 60.8 / 40.0 / 35.0 (6,464). TP8 is 12-24% below 2 x TP4 for both drafters.
**High load, 2 x TP4, 512 users per node, thinking on:** DSpark-3 13,427 tok/s total, 28.9 per user, TTFT 619 ms; DFly2-3 12,318, 26.1, 638 ms.
**Quality probe (50 prompts):** 48-49/50 for every setup.

Findings: DFly2 loses in every cell. Thinking on: -9 to -12% per user (drafter p0 0.66 vs 0.72 on V4.1 reasoning text; DFly2 released drafter trained on public chat, not V4.1 outputs). Thinking off: acceptance equal (2.53 vs 2.55) but -4 to -6% per user, from a longer step (DFly2 9.8 ms vs DSpark 9.1 ms at 20 users per copy; the drafter also reads target KV and indexer top-k). N=4 raises acceptance but costs more than it gains. Production choice stays 2 x TP4 + DSpark-3.

Issue fixed in this matrix: `vllm bench serve --chat-template-kwargs` only affects client-side prompt rendering and is never sent to the server; thinking-off needs `--extra-body '{"chat_template_kwargs":{...}}'`. First M4 rows at 20/40 users ran thinking on; moved to results/bench/invalid and M4 re-run in full (M4r).

### 16.2 DFly2 fine-tuned on V4.1 thinking-mode replies (27 Nov 2026, 13:00 to 18:57 UTC)

Goal: close DFly2's acceptance gap with thinking on by training it on V4.1's own replies.
Workspace `/data/nvme3/dfly2/ft`; runbook `REPLICATE.md` in github.com/digiland/Deepseek-h200-inference-,
branch `measured-runs`, folder `measured/dfly2/finetune`. Report: DFly2 Inference TR-02.

- Data: ShareGPT split by conversation id (14,178 held out for benchmarks; 37,456 unique training
  prompts). 20,000 replies from the production servers, thinking on, T=0.6, up to 8,192 tokens:
  53.4 M reply tokens in 52 min, 0 errors, 8.7% cut at the cap. Reasoning and answer cut from V4.1's
  own decoded tokens; AngelSpec's dsv41 template rebuilds the exact tokens for 18,042 of 18,254
  complete replies.
- Training: AngelSpec e02ea36, online on one node (V4.1 TP4 W4A8 on GPUs 4 to 7 feeding hidden states
  through Mooncake to DFly2 FSDP on GPUs 0 to 3), muon, LR 1e-4 to 1e-5, 2 epochs, 364 steps of 32
  rows x 12,288 tokens, 2.7 h. Starts from the released weights (94 tensors, 0 ignored).
- Training-data acceptance (52-step windows): first draft 0.663 -> 0.686, loss 1.431 -> 1.287.
- Held-out benchmark (2 x TP4, thinking on), tok/s per user at 40/80/160/200 users per node:
  DSpark 3 109.4/81.5/56.4/51.0; DFly2 3 released 97.2/72.7/50.7/46.0; DFly2 3 fine-tuned
  103.8/78.6/55.1/49.7; DFly2 4 fine-tuned 102.2/77.8/53.0/47.2. Probe tokens per step: 2.519 /
  2.356 / 2.513 / 2.711. Quality 48 to 49 of 50.
- Result: fine-tune +6.7 to +8.7% over the released DFly2, 96% of the acceptance gap closed, but still
  2.2 to 5.1% slower than DSpark 3: a DFly2 decode step costs 24.0 ms against 22.6 ms (reads target KV
  and indexer top-k at layers 28, 32, 36). Production stays DSpark 3.
- Fixes needed: draft config matching the release (shipped config loads 0 of 2,392 release tensors);
  release <-> training layout converters (MXFP4 requantization exact in value, 1,240 of 1,240);
  AngelSpec vLLM plugin registered through a dist-info folder; offline dump blocked by HF-name
  exporter (V4.1 uses native names), so online training; Ray socket path; wandb stand-in.
