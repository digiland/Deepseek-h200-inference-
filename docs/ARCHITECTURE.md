# Plan: 2x throughput for DeepSeek-V4.1-Flash-1017 on 4x H200

**Goal:** serve many concurrent users from one 4×H200 node at **≥ 8,000 output tok/s aggregate** (about 2,000 per GPU), with prefix caching and multimodal input kept. That must be **≥ 2× the strongest stock-vLLM baseline** on the same GPUs, not just the verbatim recipe.

**How:**
- Keep stock vLLM in Docker.
- Move from TP4 to **DP4 attention + EP4 experts** with vLLM's **SM90 MegaMoE** kernel.
- Use **DSpark speculative decoding** tuned per batch size.
- **Route each conversation to the DP rank that holds its cache.**
- Run the dense layers in **FP8 instead of BF16** (new code in `h200opt/`). This is quality-gated and is the smallest of the levers.

Then measure on the node with `scripts/ladder.sh`. SGLang runs as a cross-check.

Everything here comes from reading the code in `third_party/` at the pinned commits (vLLM `bafe5cc`, SGLang `42f4b39`, DeepInfer `a679bb8`, FlashInfer `42bbaf1`). **Nothing has been measured on H200 yet.** The numbers are estimates for the first run to confirm or kill. An independent review (§9) corrected several of them.

---

## 1. The model, as far as serving is concerned

| | Value | Why it matters |
|---|---|---|
| Layers | 40 (20 "encoder" + 20 YOCO "decoder" layers that share one KV) + 3-layer DSpark draft | Decoder layers prefill only the last ~140 tokens (bounded replay), so prefill is about 8B active params/token vs 16B for decode |
| Hidden / heads | 5120, 64 query heads, MQA, head dim 512 | vLLM's Hopper attention already runs TP-sharded heads unpadded (`nvidia/model.py:260`, `hopper/attention.py`); SGLang pads them to 64 |
| MoE | 384 routed experts, top-6, expert dim 2304, plus 1 shared; every layer | 289 GB of MXFP4 experts: 72 GB/GPU at EP4, about 92% of the weight bytes |
| Dense linears | MXFP8 (FP8 + 32×32 UE8M0 scales) | On SM90 stock vLLM dequantizes them to BF16 and adds an FP8 GEMV copy for ≤24-row steps |
| Engram | 2 hashed n-gram tables, about 197 GB FP8 | Looked up by token-id hash (about 12.7 KB/token). Lives in **host RAM** over UVA with no output change |
| Attention | 128-token SWA + compressed sparse attention (top-512 via an FP8 indexer) | KV is tiny: about 1.8 KB/token in vLLM's SM90 format, plus about 3 MB/sequence of SWA state |
| mHC | 4 residual streams with Sinkhorn mixing at every sublayer | Bandwidth-bound glue that adds **per-step fixed cost**; kernel-fused only at small batches on SM90 |
| Vision | ViT about 0.4B params | Kept; runs data-parallel |
| Checkpoint | about 510 GB in 48 shards | 564 GB of HBM, so **Engram must live on the host** |

## 2. H200 and the 4-GPU budget

See [`H200.md`](H200.md). SM90 has FP8 tensor cores (≈1,979 TFLOPS dense) but **no FP4 math**; MXFP4 experts are dequantized in-kernel. Almost all of DeepInfer's kernels are SM100-only, so DeepInfer cannot run on H200. Its engine ideas (DP+EP, DSpark, bounded replay, Engram on host) are in vLLM/SGLang with SM90 paths.

**Per-GPU memory, DP4 + EP4 + MegaMoE, Engram on host** (estimates):

| Resident on each GPU | Stock vLLM | + h200opt FP8 dense |
|---|---|---|
| Routed experts + shared expert (MegaMoE fuses the shared expert into its launch) | ~73 GB | ~73 GB |
| `wo_a` (grouped BMM, stays BF16 on SM90 either way) | ~2.7 GB | ~2.7 GB |
| Other dense linears (wq_a/wkv, wq_b, wo_b, indexer, Engram projection), full replica | ~8 GB BF16 + ~4 GB GEMV copy | ~4 GB FP8 (indexer and compressor kept BF16) |
| Embedding + LM head, ViT, DSpark draft share | ~5 GB | ~5 GB |
| Indexer logits buffer (batched tokens × max_len × FP32 = 8192 × 128K × 4) | 4 GB | 4 GB |
| MegaMoE symmetric buffers, CUDA graph pools, activations | several GB (measure) | same |
| **Left for KV at 0.92 utilization** | **~25 GB** | **~33 GB** |

At 1.8 KB/token plus 3 MB per sequence, 25 GB still holds about 13M tokens per GPU: 256 sequences at 32K context fit with room to spare. **With Engram on the host, memory does not limit batch; compute and per-step overhead do.** FP8 dense frees about 8 GB/GPU, which is nice but not decisive.

**Host RAM:** the Engram tables need about 190 GiB pinned. vLLM shares one copy across DP ranks (`dp_shared_memory` defaults on with DP>1), through host cuMem where possible and **`/dev/shm` otherwise**, which then needs about 190 GiB free on the host because `--ipc=host` uses the host's `/dev/shm`. `scripts/check_node.sh` reports it.

## 3. Where a decode step spends its time (roofline, per GPU)

T = tokens per GPU per step = sequences × (1 + draft tokens).

| Cost | Estimate |
|---|---|
| Stream this GPU's experts once per step | 72 GB / 4.8 TB/s ≈ **15 ms, fixed** |
| **Per-step fixed overhead**: mHC glue at each of 80 sublayer boundaries, sampling, DSpark drafting, scheduler, EP sync | **unknown; probably several to tens of ms**. Missing from the first draft of this plan; to be measured |
| MoE math (incl. shared) | ≈ 20 GFLOP/token → T × 33 µs at 600 TFLOPS FP8 |
| Dense math | ≈ 10 GFLOP/token. The FP8-able part (≈ 6.5 GFLOP) runs at T × 11 µs FP8 vs T × 22 µs BF16 |
| Sparse attention + indexer | ≈ 3–4 GFLOP/token at 8–32K context → T × ~6 µs |

Below roughly T≈300 a step is bound by weight bandwidth and fixed overhead. Above that, it's compute-bound at about 33 GFLOP per forward token.

**What this implies:**
- **Spec decode vs batch is roughly a wash at the roofline.**
  - No spec at 256 sequences/GPU: ≈ 15 + 256 × 0.06 ≈ 30 ms per step → about 8,500 tok/s/GPU.
  - DSpark-5 at the same 256 sequences (T = 1,536): ≈ 100 ms per step, ×3.5 accepted tokens → about 9,000 tok/s/GPU.
  - DSpark costs about 1.7 forward tokens per output token. **Its real value is amortizing the fixed per-step overhead and weight streaming.** That shows up most when sequences per GPU are limited, so whether it helps at our operating point is an empirical question. The ladder therefore sweeps both: `max` with `SPEC_SCHEDULE`, and `max-nospec` with `NODE_SEQS` up to 4096.
- **FP8 dense saves about 11 µs × T** of a step that is roughly T × 50–60 µs plus fixed costs. That is **5–10% end to end at large batch**, not the 2× its kernel speedup suggests.
- Real kernels reach maybe 25–50% of these rooflines. 2,000 tok/s/GPU (the 8,000 target) is about 25% of the roofline, plausible but not assured. **It requires several hundred concurrent sequences per GPU, and roughly 8 tok/s per user at that load.**

## 4. The levers, ranked by expected impact

| # | Lever | Mechanism | Expected | Profile / knob |
|---|---|---|---|---|
| 1 | **Batch budget**: `--max-num-seqs`, CUDA graphs covering seqs×(1+k), 128K context (shrinks the 1M-wide indexer buffer) | Keeps GPUs in the compute-bound regime. The recipe default is **128 sequences for the whole node** | Large. This alone is why the verbatim recipe is a weak baseline, hence the `tp4-tuned*` baselines | `NODE_SEQS`, all tuned profiles |
| 2 | **DSpark** (5 drafts, adaptive verification, batch-size schedule) | 1 weight pass → about 3–3.5 tokens per sequence | Large at low/mid batch, ≈neutral at the compute-bound roofline (§3) | `*-spec`, `SPEC_SCHEDULE` |
| 3 | **DP4 attention + EP4 + SM90 MegaMoE** | No per-sublayer TP all-reduces (80 per step); full-size dense GEMMs; MegaMoE fuses EP dispatch + FP8×FP4 grouped GEMM + combine and overlaps comms. vLLM ships it for SM90; the recipe enables it only on Blackwell | Medium–large at high batch. The cost: replicated dense weights, a cache split per rank, and every rank waiting for the slowest each step | `dep4-mega*`, `max*` |
| 4 | **Prefix-affinity routing across DP ranks** | vLLM's DP balancer is least-loaded per request, so turn 2 of a chat finds its cache about 25% of the time. `serve/router/affinity_router.py` pins conversations to a rank (`X-data-parallel-rank`) with bounded load | Large for multi-turn / shared-system-prompt traffic, none for one-shot prompts | `ROUTER=1` (default for DP) |
| 5 | **All-to-all backend and EPLB** | `auto` = allgather_reducescatter on H200 for the non-mega path; EPLB rebalances hot experts | Small–medium; real traffic is skewed | `EXTRA_ARGS=--all2all-backend ...`, `EPLB=1` |
| 6 | **FP8 dense GEMMs on SM90** (new: `h200opt`) | Re-blocks the 32×32 power-of-two scales into DeepGEMM SM90's 128×128 layout. Weights are exact except values pushed into FP8 subnormals, **but activations become FP8 too**, so expect ~2–3% relative error per GEMM | **5–10%** end to end, about 8 GB/GPU freed. **Quality-gated** (§7). Indexer and compressor excluded by default | `max*` |
| 7 | Rust frontend | Python (de)tokenization bottlenecks at thousands of streams | Small–medium at high concurrency | all profiles |
| 8 | SM90 threshold retuning (mHC ≤48 rows, fused all-reduce ≤12, GEMV ≤24; all tuned on H100 TP8) and fused all-reduce+mHC at TP4 | `h200opt` env knobs | Low-concurrency TP4 latency only | opt-in |

**Not pursued, and why:**
- Porting DeepInfer's SM100 kernels: weeks each, second-order gains.
- FP4→FP8 expert dequant at load: doubles the experts to about 545 GB, which doesn't fit.
- DBO / two-batch overlap: vLLM rejects DBO with Engram; SGLang rejects TBO for V4.1.

**Worth A/B-ing, not yet ranked:** DP4's 64-head decode goes through a CuTe kernel built around 8–32 heads. Compare it with `EXTRA_ARGS=--attention-config '{"backend":"FLASHMLA_SPARSE_DSV41"}'`.

## 5. Multimodal

All vLLM profiles keep the vision tower (`--mm-encoder-tp-mode data`). The `mm` scenario (one image plus text through `/v1/chat/completions`) is in the default bench set, so a regression on the image path shows up in every run. `TEXT_ONLY=1` adds `--language-model-only`.

## 6. The ladder

**Baselines.** The 2× is claimed against the best of these:

| Profile | What |
|---|---|
| `baseline` | vLLM recipe for H200, verbatim: TP4, Engram on host, 1M context, scheduler defaults (128 seqs) |
| `baseline-spec` | Same, plus the recipe's own `spec_decoding` feature |
| `tp4-tuned` | TP4 with the same batch budget as the tuned DP profiles (128K, NODE_SEQS=1024, graphs to match) |
| `tp4-tuned-spec` | `tp4-tuned` + DSpark |

**Ladder.** Each step adds one lever:

| Profile | Adds |
|---|---|
| `dep4` | DP4 + EP4 |
| `dep4-mega` | MegaMoE |
| `dep4-mega-spec` | DSpark |
| `max` | FP8 dense |
| `max-nospec` | No DSpark, swept over NODE_SEQS 1024/2048/4096 |

All DP profiles are benchmarked through the affinity router.

- **`LADDER=core`** (default, 7 entries): baseline, baseline-spec, tp4-tuned-spec, dep4-mega-spec, max, max-nospec, max-nospec at NODE_SEQS=2048.
- **`LADDER=extended`** adds per-lever attribution:
  - tp4-tuned, dep4, dep4 with alltoall_batched, dep4-mega;
  - EPLB and a spec schedule on dep4-mega-spec;
  - max-nospec at NODE_SEQS=4096;
  - SGLang baseline / dpa4 / dpa4-nospec.

**Benchmark.**
- **Headline numbers use real text:**
  - `sharegpt`: ShareGPT prompts, natural outputs up to 1k tokens.
  - `multiturn`: ShareGPT conversations replayed turn by turn with interleaving, through vLLM's multi-turn benchmark.
  - `mm`: the image path.
- `decode` (random tokens, ±30% length jitter) is kept for kernel-level A/B only. vLLM's `random` prompts are consecutive token ids, which makes DSpark acceptance, Engram and the indexer unrepresentative.
- Each point runs 6×concurrency prompts (64 to 8,192).
- Server counters are diffed before and after each run, giving per-run cache hit rate and DSpark acceptance.
- `REPEATS=3` for the numbers you quote.

**Success:**
- Peak output tok/s per headline scenario, each config at its best concurrency (`bench/summarize.py` prints peak-vs-peak).
- At least 2× vs the best baseline row.
- A passing quality gate.
- TPOT reported at the operating point.

## 7. Quality gate and risks

`bench/quality_gate.py` runs after every profile. It records:
- greedy top-5 logprobs on 64 fixed prompts;
- GSM8K accuracy (n=400, temperature 0).

Each later profile is compared with a reference (`QUALITY_REF`, default the first profile). A profile **fails** if GSM8K drops by more than 2 points, or if greedy prefix agreement falls below 30%. The agreement check is mostly informative, because TP and DP layouts already diverge numerically. FP8 throughput numbers are not to be quoted unless `max` passes with `QUALITY_REF=vllm-dep4-mega-spec`, which differs from it only by the FP8 dense path.

| Risk | Symptom | Fallback |
|---|---|---|
| SM90 MegaMoE untested on V4.1 at DP4 | Startup error naming MegaMoE / symmetric memory, or NaNs | `dep4` (Humming) stays in the extended ladder; DeepEP via `EXTRA_ARGS` |
| Adaptive verification unavailable with SM90 varlen graphs | Startup error | `ADAPTIVE=0` (SM90 then also uses the fused Markov draft sampler) |
| FP8 dense accuracy | Quality gate fails | Narrow it: `H200OPT_MXFP8_EXCLUDE='indexer\|compressor\|wq_a\|wkv'` (keep wq_b/wo_b only), or drop the plugin |
| FP8 path silently not taken | `max` ≈ `dep4-mega-spec` | Strict mode (default) fails startup if the backend is requested but unusable; the Docker build runs `python -m h200opt.check` |
| Host RAM / pinned memory | OOM or `cudaHostRegister` failure at load | Check `ulimit -l`, `df -h /dev/shm`, free RAM (`check_node.sh`) |
| DSpark hurts at the highest concurrency | `max` < `max-nospec` | `SPEC_SCHEDULE='[[1,64,5],[65,128,3],[129,1024,1]]'` |
| DP rank imbalance / expert skew | p99 TPOT ≫ p50; router `/router/stats` uneven | Router `--slack`, `EPLB=1` |
| vLLM image drift | Behaviour changes between runs | Pin the image (`docker/Dockerfile.vllm` header) |

## 8. What I need back from the first run

1. `scripts/check_node.sh > results/node_info.txt 2>&1`: topology, driver, host RAM, `/dev/shm`, memlock.
2. Inside the image: `python3 tools/reblock_report.py /model --out results/reblock.csv` (weight re-block error on the real checkpoint), and the GPU kernel test (FP8 vs BF16 speed and error per shape).
3. `scripts/ladder.sh` (core). Send `results/*/summary.md`, `results/compare_*.md`, `results/quality-*.txt`, and `results/server-*.log` for any profile that failed.

## 9. Review log

An independent review (Fable) of the first version found these problems, all fixed:

| Finding | Fix |
|---|---|
| SGLang `dpa4*` couldn't start: `--enable-dp-attention` was removed in SGLang 1.0, and `--dp-size` now means model replicas | `--attn-dp-size 4` |
| The verbatim-recipe baseline (128 seqs) made "2×" nearly automatic; `tp4-spec` changed 5 things at once | Added `baseline-spec`, `tp4-tuned`, `tp4-tuned-spec`; one node-wide `NODE_SEQS` for every profile |
| "FP8 dense is exact" was wrong (activations go FP8 too); no quality gate | Docs corrected; `bench/quality_gate.py` runs per profile; indexer/compressor excluded by default; per-1×32-group error threshold added |
| `max` could silently run BF16 | Strict mode, plus a build-time `h200opt.check` |
| Spec-decode ranking contradicted the roofline; `max-nospec` was capped at 256 seqs/rank | §3 rewritten with fixed per-step overhead; NODE_SEQS sweep to 4096; spec schedule variant |
| Bench used arange-token prompts, phase-locked lengths, cumulative metrics, and no `mm` | Real-text headline scenarios, length jitter, per-run metric deltas, repeats, `mm` in the defaults |
| DP4 splits the prefix cache with least-loaded routing | Affinity router + multi-turn scenario |
| "DP removes 16-of-64 head padding" was false for vLLM | Lever 3 rationale corrected |
| The TP4 mHC all-reduce patch was dead (name bound before the patch) | Rebinds in `nvidia/model.py` and `specialized_decode` too; verified against vLLM main on CPU |
| Minor: N%64 vs 128-row tiles; `--shm-size` ignored with `--ipc=host`; host-python deps in README | N%128 required; Docker notes; tools run inside the image |
