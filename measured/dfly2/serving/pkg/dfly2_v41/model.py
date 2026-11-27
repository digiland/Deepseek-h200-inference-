"""DFly2 drafter for DeepSeek-V4.1-Flash in vLLM 0.32.0 (SM90, FlashMLA sparse decode).

Draft layer i is a DeepSeek-V4.1 decoder layer (mHC + 128-expert MoE, reused from vLLM) whose
attention reads target layer L_i = (28, 32, 36)[i]:
  * the 128 window latents of L_i before the anchor, from L_i's sliding-window cache;
  * the global latents (KV-source layer 20's 1:1 compressed cache) at the 512 positions L_i's
    indexer picked for the last accepted token;
  * the draft block's own 6 latents, written into L_i's sliding-window cache at the block's
    positions (the next verify step rewrites those slots before it reads them).
The head adds the MTP fusion to slot 0's input and runs the hidden correction (SwiGLU plus an
attentive read of layer 36's context plus a gated prefix read of the block) slot by slot before
the shared LM head. See AngelSpec angelspec/models/{dflash.py,draft/dflash.py,draft/dsv41_*.py}.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.deepseek_v41.nvidia import model as v41_model
from vllm.models.deepseek_v41.nvidia.model import (
    DeepseekV4Model,
    _linear_scale_param_name,
    _use_sequence_parallel,
    make_deepseek_v4_expert_params_mapping,
)

logger = init_logger("vllm.dfly2_v41.model")

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")
HEAD_DIM = 512
ROPE_DIM = 64
FLASHMLA_HEADS = 64  # SM90 sparse decode takes h_q 64 or 128


def _rope_cos_sin(cos_sin_cache: torch.Tensor, positions: torch.Tensor):
    cs = cos_sin_cache.index_select(0, positions).float()
    return cs[:, : ROPE_DIM // 2], cs[:, ROPE_DIM // 2 :]


def gptj_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse=False):
    """Rotate the last 64 channels of ``x`` [T, ..., 512] in adjacent (real, imag) pairs, as
    DeepSeek-V4.1 (and AngelSpec's dsv41_mla.apply_rope) do."""
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [ROPE_DIM // 2]
    cos, sin = cos.view(shape), sin.view(shape)
    if inverse:
        sin = -sin
    tail = x[..., -ROPE_DIM:].float().unflatten(-1, (-1, 2))
    re_, im = tail[..., 0], tail[..., 1]
    rot = torch.stack([re_ * cos - im * sin, re_ * sin + im * cos], -1).flatten(-2)
    return torch.cat([x[..., :-ROPE_DIM], rot.to(x.dtype)], -1)


@dataclass
class DFlyStepContext:
    """Per draft step inputs, filled by the speculator before the forward."""

    num_reqs: int = 0
    skip_attention: bool = True
    # per draft layer (its target layer's window group): [T] block-position slots (-1 = no
    # write), [T, 1, W] window-then-block slots, [T] lengths
    slot_mapping: list = field(default_factory=list)
    swa_indices: list = field(default_factory=list)
    swa_lens: list = field(default_factory=list)
    # per draft layer: [T, 1, 512] global slots of the compressed cache, [T] lengths
    topk_indices: list = field(default_factory=list)
    topk_lens: list = field(default_factory=list)
    # correction context (layer 36, window only, no block): [R, 1, Wc], [R], [R,1,512], [R]
    corr_swa_indices: torch.Tensor | None = None
    corr_swa_lens: torch.Tensor | None = None
    corr_topk_indices: torch.Tensor | None = None
    corr_topk_lens: torch.Tensor | None = None
    # [R, hidden] target final (normed) hidden at the last accepted token
    prev_hidden: torch.Tensor | None = None
    # FlashMLA tile scheduler metas, created per step
    sched: list = field(default_factory=list)
    corr_sched: object = None
    # optional dump sink for the oracle
    dump: dict | None = None


class DFly2TargetCaches:
    """Handles to the target layers whose caches the drafter reads."""

    def __init__(self):
        self.window_attn: list = []  # target attention modules of layers 28/32/36
        self.cos_sin_cache: torch.Tensor | None = None

    def swa_cache(self, i: int) -> torch.Tensor:
        return self.window_attn[i].swa_cache_layer.kv_cache

    def compressed_cache(self, i: int) -> torch.Tensor:
        return self.window_attn[i]._compressed_kv_cache()

    def swa_block_size(self, i: int = 0) -> int:
        return self.window_attn[i].swa_cache_layer.block_size


class DFly2Attention(nn.Module):
    """Latent-MQA draft attention over a target layer's KV (no cache of its own)."""

    def __init__(self, config, draft_index: int, owner: "DFly2DeepseekV41Model", prefix: str):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        self.draft_index = draft_index
        self._owner = [owner]  # not a submodule
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // tp
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp
        self.o_lora_rank = config.o_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.eps = config.rms_norm_eps
        self.scale = HEAD_DIM**-0.5
        hidden = config.hidden_size
        self.wq_a = ReplicatedLinear(hidden, self.q_lora_rank, bias=False, quant_config=None,
                                     return_bias=False, prefix=f"{prefix}.wq_a")
        self.wkv = ReplicatedLinear(hidden, HEAD_DIM, bias=False, quant_config=None,
                                    return_bias=False, prefix=f"{prefix}.wkv")
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.kv_norm = RMSNorm(HEAD_DIM, self.eps)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * HEAD_DIM, bias=False,
                                         quant_config=None, return_bias=False,
                                         prefix=f"{prefix}.wq_b")
        self.wo_a = ColumnParallelLinear(self.n_heads * HEAD_DIM // self.n_groups,
                                         self.n_groups * self.o_lora_rank, bias=False,
                                         quant_config=None, return_bias=False,
                                         prefix=f"{prefix}.wo_a")
        self.wo_b = RowParallelLinear(self.n_groups * self.o_lora_rank, hidden, bias=False,
                                      quant_config=None, return_bias=False,
                                      prefix=f"{prefix}.wo_b")
        self.attn_sink = nn.Parameter(
            torch.full((FLASHMLA_HEADS,), -float("inf"), dtype=torch.float32),
            requires_grad=False)

    def grouped_out(self, o: torch.Tensor, cos, sin, wo_a, wo_b) -> torch.Tensor:
        """Inverse RoPE, grouped wo_a, wo_b (all-reduced unless wo_b says otherwise)."""
        T = o.shape[0]
        o = gptj_rope(o, cos, sin, inverse=True)
        o = o.reshape(T, self.n_local_groups, -1).transpose(0, 1)  # [g, T, 4096]
        w = wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)  # [g, 1024, 4096]
        z = torch.bmm(o, w.transpose(1, 2)).transpose(0, 1).reshape(T, -1)
        return wo_b(z)

    def forward(self, positions: torch.Tensor, x: torch.Tensor, _unused=None) -> torch.Tensor:
        owner = self._owner[0]
        ctx: DFlyStepContext = owner.step
        caches: DFly2TargetCaches = owner.caches
        T = x.shape[0]
        qr = self.q_norm(self.wq_a(x))
        kv = self.kv_norm(self.wkv(x))
        q = self.wq_b(qr).view(T, self.n_local_heads, HEAD_DIM)
        cos, sin = _rope_cos_sin(caches.cos_sin_cache, positions)
        if ctx.skip_attention:
            # Profiling / warm-up: exercise the GEMMs only.
            o = gptj_rope(q, cos, sin)
            return self.grouped_out(o, cos, sin, self.wo_a, self.wo_b)
        i = self.draft_index
        cache = caches.swa_cache(i)
        q_pad = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q.contiguous(), kv.contiguous(), cache.view(cache.shape[0], -1),
            ctx.slot_mapping[i][:T], positions, caches.cos_sin_cache, FLASHMLA_HEADS, self.eps,
            caches.swa_block_size(i), False, False, True, False)
        if q_pad.shape[1] != FLASHMLA_HEADS:
            q_pad = F.pad(q_pad, (0, 0, 0, FLASHMLA_HEADS - q_pad.shape[1]))
        comp = caches.compressed_cache(i)
        out, _ = flash_mla(
            q=q_pad.unsqueeze(1), k_cache=cache.unsqueeze(-2), block_table=None,
            cache_seqlens=None, head_dim_v=HEAD_DIM, tile_scheduler_metadata=ctx.sched[i],
            softmax_scale=self.scale, is_fp8_kvcache=True, indices=ctx.swa_indices[i][:T],
            topk_length=ctx.swa_lens[i][:T], attn_sink=self.attn_sink,
            extra_k_cache=comp.unsqueeze(-2), extra_indices_in_kvcache=ctx.topk_indices[i][:T],
            extra_topk_length=ctx.topk_lens[i][:T])
        o = out[:, 0, : self.n_local_heads]
        if ctx.dump is not None:
            ctx.dump[f"layer{i}.attn_in"] = x.detach().clone()
            ctx.dump[f"layer{i}.attn_o"] = o.detach().clone()
        return self.grouped_out(o, cos, sin, self.wo_a, self.wo_b)


def flash_mla(**kw):
    from vllm.v1.attention.ops.flashmla import flash_mla_with_kvcache

    return flash_mla_with_kvcache(**kw)


def new_sched():
    from vllm.v1.attention.ops.flashmla import get_mla_metadata

    return get_mla_metadata()[0]


class DFly2DecoderLayer(v41_model.DeepseekV4DecoderLayer):
    """vLLM's V4.1 decoder layer (mHC, MoE) with the attention swapped for DFly2's."""

    def __init__(self, vllm_config, prefix, draft_config, draft_index, owner, **kw):
        original = v41_model._select_dsv4_attn_cls

        def make_attn(_vllm_config):
            def build(_cfg, prefix, **_ignored):
                return DFly2Attention(draft_config, draft_index, owner, prefix)

            return build

        v41_model._select_dsv4_attn_cls = make_attn
        try:
            super().__init__(vllm_config, prefix, **kw)
        finally:
            v41_model._select_dsv4_attn_cls = original


class DFly2Correction(nn.Module):
    """TreeFlash hidden correction + DSV4.1 attentive correction with gated block keys."""

    def __init__(self, config, prefix: str):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        D = config.hidden_size
        eps = config.rms_norm_eps
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // tp
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp
        self.o_lora_rank = config.o_lora_rank
        self.scale = HEAD_DIM**-0.5
        inter = (config.dfly_config.get("hidden_correction") or {}).get("intermediate_size") or D
        self.hidden_norm = RMSNorm(D, eps)
        self.embed_norm = RMSNorm(D, eps)
        self.gate_up_proj = MergedColumnParallelLinear(2 * D, [inter, inter], bias=False,
                                                       quant_config=None, return_bias=False,
                                                       prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(inter, D, bias=False, quant_config=None,
                                           reduce_results=False, return_bias=False,
                                           prefix=f"{prefix}.down_proj")
        # attentive part
        self.q_in = ReplicatedLinear(2 * D, config.q_lora_rank, bias=False, quant_config=None,
                                     return_bias=False, prefix=f"{prefix}.attention.q_in")
        self.q_norm = RMSNorm(config.q_lora_rank, eps)
        self.wq_b = ColumnParallelLinear(config.q_lora_rank, self.n_heads * HEAD_DIM, bias=False,
                                         quant_config=None, return_bias=False,
                                         prefix=f"{prefix}.attention.wq_b")
        self.wo_a = ColumnParallelLinear(self.n_heads * HEAD_DIM // self.n_groups,
                                         self.n_groups * self.o_lora_rank, bias=False,
                                         quant_config=None, return_bias=False,
                                         prefix=f"{prefix}.attention.wo_a")
        self.wo_b = RowParallelLinear(self.n_groups * self.o_lora_rank, D, bias=False,
                                      quant_config=None, reduce_results=False,
                                      return_bias=False, prefix=f"{prefix}.attention.wo_b")
        self.kv_in = ReplicatedLinear(2 * D, HEAD_DIM, bias=False, quant_config=None,
                                      return_bias=False, prefix=f"{prefix}.attention.kv_in")
        self.kv_norm = RMSNorm(HEAD_DIM, eps)
        self.block_gate = ColumnParallelLinear(2 * D, self.n_heads, bias=False,
                                               quant_config=None, return_bias=False,
                                               prefix=f"{prefix}.attention.block_gate")
        max_slots = (config.dfly_config.get("attentive_correction") or {}).get(
            "max_block_slots", 16)
        self.attn_sink = nn.Parameter(
            torch.full((FLASHMLA_HEADS,), -float("inf"), dtype=torch.float32),
            requires_grad=False)
        self.block_sink = nn.Parameter(torch.zeros(self.n_local_heads, dtype=torch.float32),
                                       requires_grad=False)
        self.block_rel_bias = nn.Parameter(
            torch.zeros(self.n_local_heads, max_slots, dtype=torch.float32), requires_grad=False)
        self.block_gate_bias = nn.Parameter(
            torch.zeros(self.n_local_heads, max_slots, dtype=torch.float32), requires_grad=False)

    def inputs(self, hidden: torch.Tensor, prev_embeds: torch.Tensor):
        return torch.cat([self.hidden_norm(hidden), self.embed_norm(prev_embeds)], -1)

    def block_key(self, inp: torch.Tensor, cos, sin) -> torch.Tensor:
        return gptj_rope(self.kv_norm(self.kv_in(inp)), cos, sin)

    def forward_slot(self, attn: DFly2Attention, hidden, inp, keys: list, slot: int,
                     cos, sin, ctx: DFlyStepContext) -> torch.Tensor:
        """Corrected hidden for one slot. ``keys`` holds the block keys of slots 0..slot."""
        R = hidden.shape[0]
        q = self.wq_b(self.q_norm(self.q_in(inp))).view(R, self.n_local_heads, HEAD_DIM)
        q = gptj_rope(q, cos, sin)
        if ctx.skip_attention:
            read = q
        else:
            q_pad = F.pad(q, (0, 0, 0, FLASHMLA_HEADS - self.n_local_heads))
            caches = attn._owner[0].caches
            out, _ = flash_mla(
                q=q_pad.unsqueeze(1).contiguous(), k_cache=caches.swa_cache(2).unsqueeze(-2),
                block_table=None, cache_seqlens=None, head_dim_v=HEAD_DIM,
                tile_scheduler_metadata=ctx.corr_sched, softmax_scale=self.scale,
                is_fp8_kvcache=True, indices=ctx.corr_swa_indices[:R],
                topk_length=ctx.corr_swa_lens[:R], attn_sink=self.attn_sink,
                extra_k_cache=caches.compressed_cache(2).unsqueeze(-2),
                extra_indices_in_kvcache=ctx.corr_topk_indices[:R],
                extra_topk_length=ctx.corr_topk_lens[:R])
            read = out[:, 0, : self.n_local_heads]
        # gated causal read of the block's own keys 0..slot
        kb = torch.stack(keys, 1).float()  # [R, slot+1, 512]
        logits = torch.einsum("rhd,rkd->rhk", q.float(), kb) * self.scale
        back = slot - torch.arange(slot + 1, device=q.device)
        logits = logits + self.block_rel_bias[:, back].unsqueeze(0)
        total = torch.logaddexp(torch.logsumexp(logits, -1, keepdim=True),
                                self.block_sink.view(1, -1, 1))
        block_read = torch.einsum("rhk,rkd->rhd", torch.exp(logits - total), kb)
        gate = torch.tanh(self.block_gate(inp).float() + self.block_gate_bias[:, slot])
        read = (read.float() + block_read * gate.unsqueeze(-1)).to(hidden.dtype)
        attn_out = attn.grouped_out(read, cos, sin, self.wo_a, self.wo_b)
        gu = self.gate_up_proj(inp)
        g, u = gu.chunk(2, -1)
        mlp = self.down_proj(F.silu(g) * u)
        delta = tensor_model_parallel_all_reduce(attn_out + mlp)
        return hidden + delta


class DFly2DeepseekV41Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        dfly = config.dfly_config
        self.block_size = int(dfly["block_size"])
        self.hc_mult = config.hc_mult
        self.num_target_layers = config.num_hidden_layers  # 40
        self.target_kv_layers = list(dfly["target_kv_layers"])
        assert self.target_kv_layers == list(dfly["target_index_layers"])
        self.mask_token_id = int(dfly["mask_token_id"])
        self.step = DFlyStepContext()
        self.caches = DFly2TargetCaches()

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size,
                                                   prefix=f"{prefix}.embed_tokens")
        current = get_current_vllm_config()
        self.layers = nn.ModuleList([
            DFly2DecoderLayer(current, f"{prefix}.layers.{self.num_target_layers + i}",
                              config, i, self)
            for i in range(len(self.target_kv_layers))
        ])
        D = config.hidden_size
        eps = config.rms_norm_eps
        self.final_norm = RMSNorm(D, eps)
        # MTP fusion (slot0)
        self.enorm = RMSNorm(D, eps)
        self.hnorm = RMSNorm(D, eps)
        self.eh_proj = ReplicatedLinear(2 * D, D, bias=False, quant_config=None,
                                        return_bias=False, prefix=f"{prefix}.mtp_fusion.eh_proj")
        self.fusion_gate = nn.Parameter(torch.zeros(D), requires_grad=False)
        self.correction = DFly2Correction(config, f"{prefix}.hidden_correction")

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Backbone: returns final_norm(hidden) [T, D] for T = R * block_size."""
        emb = self.embed_tokens(input_ids)
        R = input_ids.shape[0] // self.block_size
        ctx = self.step
        if ctx.prev_hidden is not None and R > 0:
            emb = emb.view(R, self.block_size, -1)
            anchor = emb[:, 0]
            f = self.eh_proj(torch.cat([self.enorm(anchor),
                                        self.hnorm(ctx.prev_hidden[:R].to(anchor.dtype))], -1))
            emb = emb.clone()
            emb[:, 0] = anchor + self.fusion_gate * f
            emb = emb.view(R * self.block_size, -1)
        hidden = emb.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        residual = post_mix = res_mix = pre_mix = None
        for layer in self.layers:
            hidden, residual, post_mix, res_mix, pre_mix, _ = layer(
                hidden, positions, input_ids, pre_mix, post_mix, res_mix, residual)
        hidden = mhc_post_tilelang(hidden, residual, post_mix, res_mix)
        hidden = hc_collapse_triton(hidden, pre_mix)
        return self.final_norm(hidden)


class DFly2DeepseekV41ForCausalLM(nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.linear_scale_name = _linear_scale_param_name(
            vllm_config, getattr(self.config, "expert_dtype", "fp4"))
        self.pad_shared_expert = getattr(
            self.quant_config, "weight_block_size", None) is not None and not \
            _use_sequence_parallel(vllm_config)
        self.model = DFly2DeepseekV41Model(vllm_config=vllm_config, prefix="model")
        self.lm_head = ParallelLMHead(self.config.vocab_size, self.config.hidden_size,
                                      prefix="lm_head")
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, inputs_embeds=None):
        return self.model(input_ids, positions)

    def map_draft_to_target(self, draft_ids):
        return draft_ids  # full vocabulary

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    # ---------------- weights ----------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        first = self.model.layers[0]
        n_experts = getattr(self.config, "dspark_n_routed_experts", 0) or 128
        if first.ffn.use_native_mega_moe:
            expert_mapping = make_deepseek_v4_expert_params_mapping(n_experts)
        else:
            expert_mapping = fused_moe_make_expert_params_mapping(
                self, ckpt_gate_proj_name="w1", ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3", num_experts=n_experts)
        expert_scale_suffix = (".weight_scale" if getattr(self.config, "expert_dtype", "fp4")
                               == "fp4" else ".weight_scale_inv")
        tp = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()
        nh = self.config.num_attention_heads // tp
        h0, h1 = nh * rank, nh * (rank + 1)
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        skipped: list[str] = []
        rename = {
            "model.mtp_fusion.enorm.weight": "model.enorm.weight",
            "model.mtp_fusion.hnorm.weight": "model.hnorm.weight",
            "model.mtp_fusion.eh_proj.weight": "model.eh_proj.weight",
            "model.mtp_fusion.gate": "model.fusion_gate",
        }
        head_sliced = ("attn_sink", "block_sink", "block_rel_bias", "block_gate_bias")
        for name, w in weights:
            if name in rename:
                name = rename[name]
            elif name.startswith("model.hidden_correction."):
                rest = name[len("model.hidden_correction."):]
                if rest in ("gate_proj.weight", "up_proj.weight"):
                    p = params["model.correction.gate_up_proj.weight"]
                    p.weight_loader(p, w, 0 if rest.startswith("gate") else 1)
                    loaded.add(f"model.correction.{rest}")
                    continue
                name = "model.correction." + rest.replace("attention.", "")
            if name.endswith(".ffn.gate.bias_vl") and name not in params:
                skipped.append(name)
                continue
            # per-head vectors / tables: take this rank's heads
            if any(name.endswith(s) for s in head_sliced):
                p = params[name]
                part = w[h0:h1].float()
                p.data[: part.shape[0]].copy_(part)
                loaded.add(name)
                continue
            if name.endswith(".scale"):
                suffix = (expert_scale_suffix if _EXPERT_SCALE_RE.search(name)
                          else f".{self.linear_scale_name}")
                name = name.removesuffix(".scale") + suffix
            if ".shared_experts.w2" in name:
                name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
            if self.pad_shared_expert and ".shared_experts." in name:
                w = DeepseekV4Model._pad_shared_expert_weight(self.quant_config, name, w)
            if ".experts." in name and ".shared_experts." not in name:
                if "weight_scale" in name and w.dtype == torch.float8_e8m0fnu:
                    w = w.view(torch.uint8)
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    mapped = name.replace(weight_name, param_name)
                    p = params[mapped]
                    if p.weight_loader(p, w, mapped, shard_id=shard_id, expert_id=expert_id,
                                       return_success=True):
                        loaded.add(mapped)
                        break
                continue
            m = re.match(r"model\.layers\.\d+\.ffn\.shared_experts\.(w1|w3)\.(.*)$", name)
            if m:
                mapped = name.replace(f"shared_experts.{m.group(1)}", "shared_experts.gate_up_proj")
                p = params[mapped]
                p.weight_loader(p, w, 0 if m.group(1) == "w1" else 1)
                loaded.add(mapped)
                continue
            if name.endswith(".ffn.gate.bias"):
                name = name.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
            if name not in params:
                skipped.append(name)
                continue
            p = params[name]
            getattr(p, "weight_loader", default_weight_loader)(p, w)
            loaded.add(name)
        for layer in self.model.layers:
            layer.ffn.finalize_mega_moe_weights()
        logger.info("DFly2 drafter: loaded %d tensors (unmatched: %d %s)", len(loaded),
                    len(skipped), skipped[:8])
        if any(not s.endswith("bias_vl") for s in skipped):
            raise RuntimeError(f"DFly2 drafter: unmatched checkpoint tensors {skipped[:20]}")
        return loaded
