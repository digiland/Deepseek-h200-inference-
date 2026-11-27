# H200 inference: DeepSeek V4.1 Flash and DFly2 on 8 x NVIDIA H200

Scripts, patches, configs and results for serving DeepSeek-V4.1-Flash-1017 with vLLM 0.32.0 on one
node of 8 x H200 141 GB, and for porting, benchmarking and fine-tuning Tencent's DFly2 drafter.
No model weights and no large generated data are stored here; generated data is listed by sha256
and rebuilt by the scripts.

| Folder | Contents |
|---|---|
| `deepseek-v41/deploy` | Production recipe: 2 copies x TP4, W4A8 experts (Humming, 2 CTAs per SM), prompts admitted every 8th step, DSpark 3 drafts. `start_node.sh`, `serve_copy.sh`, `node.env`, `stop_node.sh`, benchmark and quality scripts |
| `deepseek-v41/patches` | vLLM files mounted into the container: `humming_tune` (expert GEMM tuning), `prefill_interval` (admit prompts every Nth step) |
| `dfly2/serving` | DFly2 served through vLLM 0.32.0's DSpark path: `patches` (mounted vLLM files), `pkg/dfly2_v41` (drafter code and AngelSpec reference oracle), `tools` (serve, benchmark, probe and matrix scripts), `drafter` (served config and folder builder) |
| `dfly2/finetune` | Fine-tune of DFly2 on V4.1 thinking-mode replies; runbook `REPLICATE.md` |
| `dfly2/results` | Benchmark CSVs, quality probes, probe, matrix, training and eval logs, per-step training metrics |
| `reports` | Two-page PDF reports: DeepSeek V4.1 TR-01, GLM 5.3 TR-01, DFly2 TR-01 |
| `notes-inference.md` | Working notes for every run, section by section |

Results at a glance (2 x TP4, thinking on, 80 users per node, held-out ShareGPT prompts):

| Drafter | Speed per user, tok/s | Source |
|---|---:|---|
| DSpark, 3 drafts (production) | 81.5 | DFly2 TR-02, setup E1 |
| DFly2 released, 3 drafts | 72.7 | DFly2 TR-02, setup E2 |
| DFly2 fine-tuned on V4.1 thinking replies, 3 drafts | 78.6 | DFly2 TR-02, setup E3 |
