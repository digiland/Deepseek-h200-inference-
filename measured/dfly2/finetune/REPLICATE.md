# DFly2 fine-tune on DeepSeek V4.1 Flash thinking-mode replies: replication runbook

Fine-tune of Tencent AngelSpec's DFly2 drafter (released as `AngelSlim/DS-V4.1-Flash-DFly2-B6`)
on replies that DeepSeek-V4.1-Flash-1017 itself wrote with thinking on, then served in vLLM 0.32.0
and benchmarked against V4.1's built-in DSpark drafter. Run on one node, 8 x NVIDIA H200 141 GB,
27 Nov 2026.

## 1. Inputs and versions

| Item | Value |
|---|---|
| Target model | `deepseek-ai/DeepSeek-V4.1-Flash-1017`, at `/data/nvme3/lab/teachers/deepseek-v4.1-flash-1017` |
| Released drafter | `AngelSlim/DS-V4.1-Flash-DFly2-B6` (8.2 GiB, block 6, 128 experts, 3 per token, MXFP4 experts), at `/data/nvme3/dfly2/DS-V4.1-Flash-DFly2-B6` |
| Training code | `https://github.com/Tencent/AngelSpec` commit `e02ea36e65d44e6e165c4ba73c139d500e12f2d9`, at `/data/nvme3/dfly2/AngelSpec` (not modified) |
| Container | enroot image `vllm` from `vllm-0.32.0.sqsh` (vLLM 0.32.0, torch 2.13.0+cu130, transformers 5.19.0, Mooncake included); `. /data/nvme3/shona-cpt/enroot/env.sh` sets up enroot |
| Extra Python packages | `support/pydeps-versions.txt`, installed with `pip install --no-deps --target /data/nvme3/dfly2/ft/pydeps ...` inside the container; container site-packages untouched |
| Prompts | ShareGPT V3 unfiltered cleaned split, `/data/nvme3/lab/teachers/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json` |

All scripts use absolute paths under `/data/nvme3`; scripts live in `/data/nvme3/dfly2/ft` on the node
(copies here). `sha256` of every generated data file: `data_sha256.txt`.

## 2. Steps

### 2.1 Split ShareGPT: held-out benchmark set and training prompts

```
python3 split_sharegpt.py
```

Conversation id md5 mod 100 < 15 goes to `data/sharegpt_heldout.json` (14,178 conversations, used
only for benchmarks); the rest gives 37,456 unique first-turn prompts (20 to 6,000 characters),
shuffled with seed 7, in `data/train_prompts.jsonl`.

### 2.2 Generate V4.1 thinking-mode replies (about 52 min)

Start the production servers (2 copies x TP4, W4A8, DSpark 3 drafts) on ports 8180 and 8181:

```
PORT=8180 GPUS=0,1,2,3 TP=4 MODE=dspark NSPEC=3 REJ=block DRAFT_SAMPLE=probabilistic bash ../serving/tools/serve_v41_any.sh &
PORT=8181 GPUS=4,5,6,7 TP=4 MODE=dspark NSPEC=3 REJ=block DRAFT_SAMPLE=probabilistic bash ../serving/tools/serve_v41_any.sh &
./run_gen.sh      # gen_replies.py: first 20,000 prompts, T=0.6, top_p=0.95, max 8,192 tokens, thinking on
```

Result: 20,000 replies, 0 errors, 53.4 million reply tokens, 17,000 tok/s; 8.7% cut at 8,192 tokens.
`gen_replies.py` keeps the exact prompt and reply token ids (`return_token_ids`). Stop the servers.

### 2.3 Convert to AngelSpec conversations

```
python3 to_conversations.py data/replies_v1.jsonl data/train_v1.jsonl 300
```

The reasoning and answer are cut from V4.1's own decoded reply at `</think>` (not taken from
vLLM's reasoning parser, which strips a space before `</think>`). Check printed: AngelSpec's
`dsv41` template rebuilds the exact token ids for 300 of 300 checked samples (cut replies get 2
closing tokens added by the template).

### 2.4 Prepare the drafter for training

```
python3 release_to_train.py /data/nvme3/dfly2/DS-V4.1-Flash-DFly2-B6 init/release_train_layout
```

Writes the release in AngelSpec training layout (94 tensors; MXFP4 experts dequantized and
stacked to bf16) and `init/draft_config_ft.json` (copy here: `draft_config_ft.json`). The draft
config differs from AngelSpec's shipped `dfly2_dsv41_flash_draft_config.json` in the settings the
release was actually trained with: `hidden_correction_intermediate_size 5120`,
`attentive_correction_block_keys true`, `attentive_correction_block_mode gated`,
`attentive_correction_block_kv_in hidden_embed`, `enable_confidence_head false`. With the shipped
config the trainer loads 0 of 2,392 released tensors and silently trains from random init.

Check the reverse path (training layout back to serving layout) before training:

```
python3 train_to_release.py init/release_train_layout /data/nvme3/dfly2/served/DS-V4.1-Flash-DFly2-B6 init/roundtrip
python3 check_roundtrip_values.py /data/nvme3/dfly2/DS-V4.1-Flash-DFly2-B6 init/roundtrip
# expected: weights compared 1240: identical values 1240, different 0
```

### 2.5 Register AngelSpec's vLLM plugin without installing AngelSpec

AngelSpec applies its vLLM runtime patches (including the DeepSeek V4.1 attention-input capture
that DFly2 training needs) through the `vllm.general_plugins` entry point, which exists only when
the package is installed. Copy `support/angelspec-0.0.0+local.dist-info` into the `pydeps`
directory on `PYTHONPATH`. Copy `support/stubs` (no-op `wandb` and `swanlab`; logging is off) to
`/data/nvme3/dfly2/ft/stubs`.

### 2.6 Train (online, one node, 2 h 41 min)

```
./run_train.sh /data/nvme3/dfly2/ft/data/train_v1.jsonl ft_v1 \
  training.num_epochs=2 training.draft_accumulation_steps=8 \
  training.learning_rate=1e-4 training.min_lr=1e-5 \
  training.save_interval=50 training.max_checkpoints=3
```

`run_train.sh` runs AngelSpec's `configs/vllm_dsv41_flash_dfly2.yaml` with: V4.1 inference TP4 on
GPUs 4 to 7 (W4A8 like serving), DFly2 training FSDP on GPUs 0 to 3, image input off,
`max_seq_length 12288`, `draft_init_path` = the converted release, `draft_model_config` = the
matching config. Resolved config: `train_ft_v1_resolved_config.yaml`. 364 steps, 32 packed rows
of up to 12,288 tokens per step. V4.1 prefill sets the pace (about 18,000 to 30,000 tokens per
second fed to training).

Offline capture (`run_capture.sh`, AngelSpec `online_dump`) does not work for this checkpoint:
AngelSpec's frozen-asset exporter expects Hugging Face key names (`model.embed_tokens.weight`,
`self_attn.kv_a_proj_with_mqa`) and the V4.1 checkpoint uses native names (`embed.weight`,
`layers.N.attn.wkv`). Online training avoids the exporter.

### 2.7 Export and serve

```
python3 train_to_release.py runs/ft_v1/dfly2_dsv41_flash/hf_final /data/nvme3/dfly2/served/DS-V4.1-Flash-DFly2-B6 served/DFly2-ft-v1
DFLY2_MODEL=/data/nvme3/dfly2/ft/served/DFly2-ft-v1 MODE=dfly2 NSPEC=3 REJ=standard ... bash ../serving/tools/serve_v41_any.sh
```

MXFP4 requantization: e2m1 values, one power-of-two (E8M0) scale per 32 inputs, chosen so that
the block maximum lands in (3, 6] times the scale, as in the release.

### 2.8 Evaluate on the held-out prompts (about 1 h 40 min)

```
./run_ft_eval.sh /data/nvme3/dfly2/ft/served/DFly2-ft-v1
```

2 copies x TP4, thinking on, T=0.6, top_p 0.95, output capped at 256, 16 x users requests per
rung, seeds 71000 and 72000, 20/40/80/100 users per copy. Per setup: quality probe (50 questions),
acceptance probe (160 prompts), benchmark. `chain_after_train.sh` runs 2.7 and 2.8 automatically
after training.

## 3. Results

Training metrics per step: `../results/train_ft_v1_metrics.csv`. Averages over 52-step windows:

| Steps | Loss | First draft accepted | Accepted length (trainer) |
|---|---:|---:|---:|
| 1 to 52 | 1.431 | 0.663 | 2.004 |
| 157 to 208 | 1.325 | 0.679 | 2.098 |
| 313 to 364 | 1.287 | 0.686 | 2.140 |

Held-out benchmark, speed per user in tok/s (2 x TP4, thinking on):

RESULTS_TABLE

Raw rows: `../results/data/bench/results.csv` (tags `E1_` to `E4_`); log: `../results/logs/eval.log`.
