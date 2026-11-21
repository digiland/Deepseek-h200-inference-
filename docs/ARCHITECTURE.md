# Plan: 2x throughput for DeepSeek-V4.1-Flash-1017 on 4x H200

**Goal:** serve many concurrent users from one 4×H200 node at **≥ 8,000 output tok/s aggregate** (about 2,000 per GPU), with prefix caching and multimodal input kept. That should be at least 2× the stock vLLM recipe on the same GPUs.

**How it gets there:** keep stock vLLM (Docker) and change four things.
1. Switch from TP4 to **DP4 attention + EP4 experts**.
2. Turn on the **SM90 MegaMoE** kernel vLLM already ships.
3. Run the dense layers in **FP8 instead of BF16**. This is new code in `h200opt/`.
4. Use **DSpark speculative decoding**, with draft length tuned by batch size.

Then measure each step on the node with `scripts/ladder.sh`. SGLang runs in parallel as a cross-check.

Everything below comes from reading the code in `third_party/` at the pinned commits (vLLM `bafe5cc`, SGLang `42f4b39`, DeepInfer `a679bb8`, FlashInfer `42bbaf1`). Nothing has been measured on H200 yet; the numbers are estimates to check against the first run.

---

## 1. The model, as far as serving is concerned

| | Value | Why it matters |
|---|---|---|
| Layers | 40 (20 "encoder" + 20 YOCO "decoder" layers sharing one KV) + 3-layer DSpark draft | Decoder layers prefill only the last ~140 tokens (bounded replay), so prefill costs about 8B active params per token instead of 16B |
| Hidden / heads | 5120, 64 query heads, MQA (1 KV head), head dim 512 | At TP4 each rank gets 16 heads; FlashMLA pads to 64 (4× waste) unless handled |
| MoE | 384 routed experts, top-6, expert dim 2304, plus 1 shared; every layer | 289 GB of MXFP4 experts. That's 72 GB per GPU at EP4 and 92% of the weight bytes |
| Dense linears | ≈7B params, MXFP8 (FP8 + 32×32 UE8M0 scales) | **On SM90 stock vLLM dequantizes these to BF16** (2× bytes, ½ tensor throughput) |
| Engram | 2 hashed n-gram tables, about 197 GB FP8 | Looked up by token-id hash, about 12.7 KB per token. Lives in **host RAM** over UVA with no output change |
| Attention | 128-token SWA + compressed sparse attention (top-512 of 2× or 1× compressed KV via an FP8 indexer) | KV is tiny: **≈1.8 KB/token** in vLLM's SM90 format, plus ≈3 MB/sequence of SWA state |
| mHC | 4 residual streams (4×5120) with Sinkhorn mixing at every sublayer | Bandwidth-bound glue that is kernel-fused only at small batches on SM90 |
| Vision | ViT ≈0.4B params | Small. Kept on, running data-parallel (`--mm-encoder-tp-mode data`) |
| Checkpoint | ≈510 GB in 48 shards | 4×141 = 564 GB of HBM, so **Engram must go to host memory** |

## 2. H200 and the 4-GPU budget

See [`H200.md`](H200.md). The key facts:
- SM90 has FP8 tensor cores (≈1,979 TFLOPS dense) but **no FP4 math**. MXFP4 experts are dequantized inside the kernel (W4A8 → FP8 wgmma, or W4A16 → BF16).
- HBM is 4.8 TB/s per GPU. NVLink/NVSwitch gives 450 GB/s each way between any pair of GPUs.
- Almost all of DeepInfer's kernels are **SM100-only**, so DeepInfer cannot run on H200. Its engine ideas (DP+EP, DSpark, bounded replay, Engram on host) are already in vLLM/SGLang with SM90 code paths.

**Per-GPU memory, DP4+EP4 (TP1), Engram on host** (estimates):

| Resident on each GPU | Stock vLLM | With h200opt FP8 dense |
|---|---|---|
| Routed experts (1/4 of 289 GB, MXFP4) | 72 GB | 72 GB |
| Dense linears (full replica in DP) | ~14 GB BF16 + ~7 GB FP8 GEMV copy | **~7 GB FP8** |
| Embedding + LM head, ViT, DSpark draft share | ~5 GB | ~5 GB |
| **Weights total** | **~98 GB** | **~84 GB** |
| Left for KV and activations at 0.92 utilization | ~31 GB | ~45 GB |

At 1.8 KB/token plus 3 MB per sequence, 31–45 GB holds **17–25 million tokens per GPU**. That's 256 sequences at 64K context with room to spare. **Once Engram is on the host, memory is not what limits batch size; compute is.**

**Host RAM:** the Engram tables need about 190 GiB pinned. vLLM shares one copy across DP ranks on a node (`dp_shared_memory` defaults on with DP>1). Budget at least 256 GB free host RAM. The container runs with `--ipc=host` and `memlock=-1`.

## 3. Where a decode step spends its time (roofline, per GPU)

A step processes T tokens per GPU, where T is sequences × (1 + draft tokens).

| Cost | Size | Time |
|---|---|---|
| Stream this GPU's experts once per step | 72 GB / 4.8 TB/s | **~15 ms**, fixed |
| MoE math | 425 MFLOP/token/layer × 40 = 17 GFLOP/token | T × 28 µs at ~600 TFLOPS FP8 |
| Dense math (attention projections + shared expert) | 13 GFLOP/token | T × 22 µs FP8, **T × 44 µs BF16 (stock SM90)** |
| Sparse attention + indexer | ≈3–4 GFLOP/token at 8–32K context | T × ~6 µs |

- Below about T≈300 the step is **bandwidth-bound**: batch more and you get free tokens.
- Above that it is **compute-bound** at about 50 GFLOP per forward token. In that regime the dense layers are about 40% of the math, and stock vLLM runs them in BF16. **That is what the FP8 dense path fixes.**

For example, at T=1536 (256 sequences × 6 with DSpark-5):
- FP8: ≈15 + 43 + 34 + 9 ≈ **100 ms/step**. At an acceptance length of 3.5 that is 256 × 3.5 / 0.1 ≈ **9,000 output tok/s per GPU** at the roofline.
- Stock BF16 dense: ≈ 134 ms/step.

Real kernels reach maybe 25–50% of the roofline. That still leaves 2,000+ tok/s per GPU, which is the 8,000 aggregate target, within reach. It requires running **hundreds of concurrent sequences per GPU**.

## 4. The levers, ranked

| # | Lever | Where | Expected effect | Status |
|---|---|---|---|---|
| 1 | **DP4 attention + EP4 experts** instead of TP4 | vLLM flags | No per-layer TP all-reduces, no 16-of-64 head padding, attention batch ×4 per GPU. Makes large batches cheap | `dep4` profile |
| 2 | **DeepGEMM SM90 MegaMoE**: fused EP dispatch + FP8×FP4 grouped GEMM + combine over NVLink symmetric memory | vLLM `--kernel-config '{"moe_backend":"deep_gemm_mega_moe"}'` (ships in main for SM90; the recipe enables it only on Blackwell) | Removes separate all-to-all kernels and overlaps communication with GEMM. Uses FP8 tensor cores | `dep4-mega` |
| 3 | **DSpark speculative decoding** (5 drafts, adaptive verification) | vLLM flags (off by default) | 1 weight pass gives about 3–3.5 tokens per sequence. Biggest win at low and mid batch; at very high batch verify compute eats into it, so the profile can use a batch-size schedule (`SPEC_SCHEDULE`) | `dep4-mega-spec` |
| 4 | **Native FP8 dense GEMMs on SM90** | **New: `h200opt` vLLM plugin** | 2× tensor throughput on about 40% of large-batch FLOPs, frees about 14 GB/GPU (no BF16 copy, no GEMV copy). At load, the 32×32 power-of-two scales are re-blocked into the 128×128 layout DeepGEMM SM90 takes. This is exact except for values pushed into FP8 subnormals; any layer above 0.5% error keeps BF16 | `max` |
| 5 | Context window sized to the workload (`--max-model-len 131072` instead of 1M) | flags | The indexer allocates a [batched tokens × max_len] FP32 logits buffer: 32 GB at 1M, 4 GB at 128K. That headroom becomes batch | all tuned profiles |
| 6 | Bigger scheduler budget and CUDA graphs that cover it (`max-num-seqs 256/rank`, graphs up to seqs × (1+k)) | flags | Keeps the GPU in the compute-bound regime without eager fallbacks | all tuned profiles |
| 7 | Prefix caching, with bounded SWA replay | default on | Repeated system prompts and multi-turn chats skip prefill; the SWA window is recomputed (128 tokens) | all profiles |
| 8 | Rust frontend (`VLLM_USE_RUST_FRONTEND=1`) | env | Python tokenization and detokenization become a bottleneck at thousands of streams | all profiles |
| 9 | SM90 threshold retuning (mHC kernels ≤48 rows, fused all-reduce ≤12 rows, GEMV ≤24 rows; all tuned on H100 TP8) and fused all-reduce+mHC at TP4 | `h200opt` env knobs | Low-concurrency latency only; TP4 profiles | opt-in |

**Not pursued, and why:**
- Porting DeepInfer's SM100 kernels: FlashMLA's new KV format, MXFP4 indexer, mega-mHC. These are weeks of kernel work each, and the gains (5–15%) are second-order next to 1–4.
- FP4 → FP8 expert dequant at load (SGLang `SGLANG_DSV4_FP4_DEQUANT`): doubles the experts to about 545 GB, which does not fit.
- DBO / two-batch overlap: vLLM rejects DBO with Engram, and SGLang rejects TBO for V4.1.

## 5. Multimodal

All vLLM profiles keep the vision tower (`--mm-encoder-tp-mode data` puts a full ViT replica on each rank, and the ViT is about 0.4B params). `TEXT_ONLY=1` adds `--language-model-only` if you later want the last GB back. The `mm` benchmark scenario exercises the image path through `/v1/chat/completions`.

## 6. The A/B ladder (what to run, in order)

Each profile adds one lever on top of the previous one, so the results attribute the gain:

| Profile | Adds |
|---|---|
| `baseline` | vLLM recipe for H200, verbatim: TP4 + Engram on host, model-default 1M context, no spec decode |
| `tp4-spec` | 128K context, bigger batch, DSpark |
| `dep4` | DP4 attention + EP4 (default SM90 MoE = Humming W4A8) |
| `dep4-mega` | SM90 MegaMoE |
| `dep4-mega-spec` | DSpark |
| `max` | FP8 dense GEMMs (plugin) |
| `max-nospec` | `max` without DSpark, for the highest-concurrency point |

SGLang cross-check: `sglang:baseline` (InferenceX H200 recipe), `sglang:dpa4` (DP attention + EP4 + W4A8 MoE + DSpark), `sglang:dpa4-nospec`, `sglang:dpa4-hicache`.

**Success metric:** peak output tok/s per scenario, with each config at its best concurrency (`bench/summarize.py` prints peak-vs-peak). Also check TPOT at the operating point; 8,000 tok/s at 1,024 concurrent streams is about 8 tok/s per user.

## 7. Risks and what I'll do if they hit

| Risk | Symptom | Fallback |
|---|---|---|
| SM90 MegaMoE path untested on V4.1 at DP4 | startup error naming MegaMoE/symmetric memory, or NaNs | `dep4` (Humming) stays in the ladder; also try `EXTRA_ARGS="--all2all-backend deepep_high_throughput"` |
| Adaptive verification not available on SM90 varlen graphs | startup error mentioning adaptive verification / CUDA graph support | `ADAPTIVE=0` (SM90 then also gets the fused Markov draft sampler) |
| FP8 dense accuracy | `tools/reblock_report.py` shows layers above threshold, or a GSM8K drop | per-layer BF16 fallback is automatic; tighten `H200OPT_MXFP8_MAX_REL_ERR` or drop the plugin |
| Host RAM / pinned-memory limits | OOM or `cudaHostRegister` failures at load | `--engram-config '{"cpu_offload":true,"dp_shared_memory":true}'`, more `--shm-size`, check `ulimit -l` |
| DSpark hurts at the highest concurrency | `max` below `max-nospec` at c≥1024 | `SPEC_SCHEDULE='[[1,64,5],[65,192,3],[193,1024,1]]'` |
| Docker `nightly` image drifts | different behaviour between runs | pin the image digest after the first good run |

## 8. What I need back from the first run

1. `scripts/check_node.sh > results/node_info.txt`: topology, driver, host RAM.
2. `python tools/reblock_report.py $MODEL_DIR --out results/reblock.csv`: FP8 dense accuracy on the real weights. Takes minutes, CPU only.
3. `H200OPT_MXFP8_DEEPGEMM=1 python -m pytest -q h200opt/tests/gpu -s` inside the vLLM image, on one GPU: kernel speed vs BF16 cuBLAS per shape.
4. `scripts/ladder.sh` (≈ 1 hour per profile including load), then send `results/*/summary.md`, `results/compare_*.md` and the server logs of any profile that failed.
