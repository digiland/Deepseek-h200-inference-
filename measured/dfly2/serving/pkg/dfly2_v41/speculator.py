"""DFly2 speculator for DeepSeek-V4.1-Flash (vLLM 0.32.0 model runner V2).

One draft step per verify step: a parallel backbone pass over a block of 6 queries per request
(anchor + 5 masks, positions a..a+5) and then a sequential head over slots 1..N (N =
num_speculative_tokens <= 5), each slot corrected with the previous slot's sampled token.
"""

import os
import re
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from . import state
from .model import DFly2DeepseekV41ForCausalLM, gptj_rope, new_sched, _rope_cos_sin

logger = init_logger("vllm.dfly2_v41.speculator")

WINDOW = 128
SWA_WIDTH = 192  # window + block, padded to 64
CORR_WIDTH = 128
TOPK = 512


@triton.jit
def _dfly_swa_kernel(
    swa_idx_ptr, swa_len_ptr, corr_swa_ptr, corr_swa_len_ptr, anchor_pos_ptr,
    swa_bt_ptr, swa_bt_stride, swa_bs,
    WINDOW: tl.constexpr, BLOCK: tl.constexpr, WS: tl.constexpr, WC: tl.constexpr,
    PW: tl.constexpr, WRITE_CORR: tl.constexpr,
):
    """Window [a-128, a) then block [a, a+BLOCK) slots of one target layer's window group."""
    r = tl.program_id(0)
    a = tl.load(anchor_pos_ptr + r)
    nw = tl.minimum(a, WINDOW)
    i = tl.arange(0, PW)
    is_win = i < nw
    is_blk = (i >= nw) & (i < nw + BLOCK)
    pos = tl.where(is_win, a - nw + i, a + (i - nw))
    use = is_win | is_blk
    bid = tl.load(swa_bt_ptr + r * swa_bt_stride + pos // swa_bs, mask=use, other=0).to(tl.int64)
    ok = use & (bid != 0)
    slot = tl.where(ok, bid * swa_bs + pos % swa_bs, -1).to(tl.int32)
    n_ok = tl.sum(ok.to(tl.int32), axis=0)
    for j in range(BLOCK):
        q = r * BLOCK + j
        tl.store(swa_idx_ptr + q * WS + i, slot, mask=i < WS)
        tl.store(swa_len_ptr + q, n_ok)
    if WRITE_CORR:
        cslot = tl.where(is_win & ok, slot, -1)
        tl.store(corr_swa_ptr + r * WC + i, cslot, mask=i < WC)
        tl.store(corr_swa_len_ptr + r, tl.sum((is_win & ok).to(tl.int32), axis=0))


@triton.jit
def _dfly_topk_kernel(
    topk_out_ptr, topk_len_ptr, corr_topk_ptr, corr_topk_len_ptr, anchor_row_ptr,
    comp_bt_ptr, comp_bt_stride, comp_bs,
    topk0_ptr, topk1_ptr, topk2_ptr, topk_stride, num_rows_total,
    BLOCK: tl.constexpr, TOPK: tl.constexpr,
):
    """Global compressed-cache slots of each target layer's top-k row for the last accepted
    token, repeated for the request's BLOCK queries."""
    r = tl.program_id(0)
    row = tl.load(anchor_row_ptr + r)
    k = tl.arange(0, TOPK)
    for layer in range(3):
        if layer == 0:
            src = topk0_ptr
        elif layer == 1:
            src = topk1_ptr
        else:
            src = topk2_ptr
        local = tl.load(src + row * topk_stride + k)
        valid = local >= 0
        cb = tl.load(comp_bt_ptr + r * comp_bt_stride + local // comp_bs, mask=valid,
                     other=0).to(tl.int64)
        g = tl.where(valid, cb * comp_bs + local % comp_bs, -1).to(tl.int32)
        cnt = tl.sum(valid.to(tl.int32), axis=0)
        for j in range(BLOCK):
            q = r * BLOCK + j
            tl.store(topk_out_ptr + (layer * num_rows_total + q) * TOPK + k, g)
            tl.store(topk_len_ptr + layer * num_rows_total + q, cnt)
        if layer == 2:
            tl.store(corr_topk_ptr + r * TOPK + k, g)
            tl.store(corr_topk_len_ptr + r, cnt)


class DFly2Speculator(DSparkSpeculator):
    _speculator_name = "DFly2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        hf = self.draft_model_config.hf_config
        dfly = hf.dfly_config
        self.block = int(dfly["block_size"])
        assert self.num_speculative_steps <= self.block - 1, "num_speculative_tokens <= 5"
        self.full = self.block - 1
        self.num_query_per_req = self.block
        self.mask_token_id = int(dfly["mask_token_id"])
        self.target_layers = list(dfly["target_kv_layers"])
        R = self.max_num_reqs
        T = R * self.block
        dev = device
        self._anchor_idx = torch.arange(R, dtype=torch.int64, device=dev) * self.block
        self.q_slot_mapping = torch.full((3, self.max_num_tokens + T), -1, dtype=torch.int64,
                                         device=dev)
        self._ctx_pos = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=dev)
        self._ctx_slot = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=dev)
        self.s_indices = torch.zeros(R * self.full, dtype=torch.int64, device=dev)
        self.s_pos = torch.zeros(R * self.full, dtype=torch.int64, device=dev)
        self.s_idx_map = torch.full((R * self.full,), -1, dtype=torch.int64, device=dev)
        i32 = torch.int32
        self.swa_idx = torch.full((3, T, SWA_WIDTH), -1, dtype=i32, device=dev)
        self.swa_len = torch.zeros(3, T, dtype=i32, device=dev)
        self.topk_idx = torch.full((3, T, TOPK), -1, dtype=i32, device=dev)
        self.topk_len = torch.zeros(3, T, dtype=i32, device=dev)
        self.corr_swa_idx = torch.full((R, CORR_WIDTH), -1, dtype=i32, device=dev)
        self.corr_swa_len = torch.zeros(R, dtype=i32, device=dev)
        self.corr_topk_idx = torch.full((R, TOPK), -1, dtype=i32, device=dev)
        self.corr_topk_len = torch.zeros(R, dtype=i32, device=dev)
        self.prev_hidden = torch.zeros(R, hf.hidden_size, dtype=self.dtype, device=dev)
        self.draft_top_p = float(os.environ.get("DFLY2_DRAFT_TOP_P", "1.0"))
        self.draft_topk_cand = int(os.environ.get("DFLY2_DRAFT_TOPK", "256"))
        self.dump_dir = os.environ.get("DFLY2_DUMP_DIR")
        self.dump_steps = int(os.environ.get("DFLY2_DUMP_STEPS", "0"))
        self.dump_every = int(os.environ.get("DFLY2_DUMP_EVERY", "1"))
        self.dump_min_reqs = int(os.environ.get("DFLY2_DUMP_MIN_REQS", "1"))
        self._dumped = 0
        self._step = 0
        self.stats_steps = 0

    # ------------------------------------------------------------------ loading
    def load_draft_model(self, target_model: nn.Module, target_attn_layer_names: set[str]):
        import vllm.model_executor.model_loader.base_loader as base_loader
        from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

        original = base_loader.initialize_model

        def init_dfly(**kw):
            kw["model_class"] = DFly2DeepseekV41ForCausalLM
            return original(**kw)

        base_loader.initialize_model = init_dfly
        try:
            model = load_dspark_model(target_model, self.vllm_config)
        finally:
            base_loader.initialize_model = original
        assert isinstance(model, DFly2DeepseekV41ForCausalLM), type(model)
        self.use_confidence_head = False
        self._bind_target_layers(model)
        return model

    def _bind_target_layers(self, model) -> None:
        ctx = self.vllm_config.compilation_config.static_forward_context
        attn_by_layer = {}
        for name, mod in ctx.items():
            m = re.search(r"layers\.(\d+)\.attn$", name)
            if m and hasattr(mod, "swa_cache_layer"):
                attn_by_layer[int(m.group(1))] = mod
        caches = model.model.caches
        caches.window_attn = [attn_by_layer[L] for L in self.target_layers]
        caches.cos_sin_cache = caches.window_attn[0].rotary_emb.cos_sin_cache
        for mod in caches.window_attn:
            assert mod.compress_ratio == 1 and mod.kv_source_layer_id == 20, mod.prefix

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        self.query_cudagraph_manager = None
        self.use_graphs = (os.environ.get("DFLY2_CUDAGRAPH", "1") == "1"
                           and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL)
        sizes = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 144, 160,
                 192, 224, 256)
        self.graph_sizes = [s for s in sizes if s <= self.max_num_reqs]
        self.graphs: dict = {}

    def _graph_inputs(self) -> None:
        """Inert inputs for capture: no KV writes, empty reads, inert sampling rows."""
        T = self.max_num_reqs * self.block
        self.input_buffers.input_ids[:T].fill_(self.mask_token_id)
        self.input_buffers.positions[:T].copy_(torch.arange(T, device=self.device) % self.block + 1)
        self.q_slot_mapping.fill_(-1)
        self.swa_len.zero_()
        self.topk_len.zero_()
        self.corr_swa_len.zero_()
        self.corr_topk_len.zero_()
        self.s_idx_map.fill_(-1)
        self.s_pos.zero_()
        self.prev_hidden.zero_()

    def _static_ctx(self, R: int) -> None:
        ctx = self.model.model.step
        ctx.skip_attention = False
        ctx.num_reqs = R
        ctx.slot_mapping = [self.q_slot_mapping[i] for i in range(3)]
        ctx.swa_indices = [self.swa_idx[i].unsqueeze(1) for i in range(3)]
        ctx.swa_lens = [self.swa_len[i] for i in range(3)]
        ctx.topk_indices = [self.topk_idx[i].unsqueeze(1) for i in range(3)]
        ctx.topk_lens = [self.topk_len[i] for i in range(3)]
        ctx.corr_swa_indices = self.corr_swa_idx.unsqueeze(1)
        ctx.corr_swa_lens = self.corr_swa_len
        ctx.corr_topk_indices = self.corr_topk_idx.unsqueeze(1)
        ctx.corr_topk_lens = self.corr_topk_len
        ctx.prev_hidden = self.prev_hidden
        ctx.dump = None

    @torch.inference_mode()
    def capture(self) -> None:
        if not self.use_graphs:
            logger.info("DFly2: draft step runs eagerly.")
            return
        from vllm.distributed.parallel_state import graph_capture
        from vllm.platforms import current_platform

        pool = current_platform.get_global_graph_pool()
        ctx = self.model.model.step
        self._graph_inputs()
        with graph_capture(device=self.device) as gc:
            for R in reversed(self.graph_sizes):
                self._static_ctx(R)
                ctx.sched = [new_sched() for _ in range(3)]
                ctx.corr_sched = new_sched()
                self._draft(R, sample=True)  # warm-up
                torch.cuda.synchronize()
                scheds = [new_sched() for _ in range(3)]
                corr = new_sched()
                ctx.sched, ctx.corr_sched = scheds, corr
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool, stream=gc.stream):
                    self._draft(R, sample=True)
                self.graphs[R] = (g, scheds, corr)
        torch.cuda.synchronize()
        self._graph_inputs()
        logger.info("DFly2: captured %d draft-step CUDA graphs (%s requests)",
                    len(self.graphs), self.graph_sizes)

    def set_attn(self, model_state, kv_cache_config, block_tables, target_input_buffers,
                 target_attn_groups) -> None:
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        self.target_input_buffers = target_input_buffers
        self.target_attn_groups = target_attn_groups
        self.attn_groups = []
        caches = self.model.model.caches
        name_to_gid = {ln: g for g, grp in enumerate(kv_cache_config.kv_cache_groups)
                       for ln in grp.layer_names}
        self.swa_gids = [name_to_gid[m.swa_cache_layer.prefix] for m in caches.window_attn]
        self.swa_bss = [int(block_tables.block_sizes[g]) for g in self.swa_gids]
        comp_name = caches.window_attn[0].compressed_cache_prefix
        self.comp_gid = name_to_gid[comp_name]
        self.comp_bs = int(block_tables.block_sizes[self.comp_gid])
        for i, g in enumerate(self.swa_gids):
            assert self.swa_bss[i] == caches.swa_block_size(i), (self.swa_bss, i)
        for L in self.target_layers:
            assert L in state.TOPK_BUFFERS, (
                f"target layer {L} publishes no top-k copy; set DFLY2_TOPK_LAYERS and mount "
                "the patched attention.py")
        logger.info("DFly2: window groups %s (blocks %s), compressed group %d (block %d) from %s",
                    self.swa_gids, self.swa_bss, self.comp_gid, self.comp_bs, comp_name)

    # ------------------------------------------------------------------ drafting
    @torch.inference_mode()
    def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states,
                aux_hidden_states, num_sampled, num_rejected, last_sampled, next_prefill_tokens,
                temperature, seeds, dp_sync_state=None, dummy_run=False,
                skip_attn_for_dummy_run=False, mm_inputs=None, is_profile=False,
                num_speculative_tokens=None) -> torch.Tensor:
        R = input_batch.num_reqs
        B = self.block
        T = R * B
        model = self.model.model
        ctx = model.step
        if dummy_run:
            self.input_buffers.input_ids[:T].fill_(self.mask_token_id)
            self.input_buffers.positions[:T].copy_(
                torch.arange(T, device=self.device) % B + 1)
            ctx.skip_attention = True
            ctx.prev_hidden = self.prev_hidden
            self.prev_hidden[:R].zero_()
            self._draft(R, sample=False)
            return self.draft_tokens[:R]

        bt = self.block_tables.input_block_tables
        for i, g in enumerate(self.swa_gids):
            # Same outputs for every group except the block-position slot mapping.
            prepare_dflash_inputs(
                self.input_buffers, self.q_slot_mapping[i], self._ctx_pos, self._ctx_slot,
                self.s_indices, self.s_pos, self.s_idx_map, self.temperature, self.seeds,
                input_batch, num_sampled, num_rejected, last_sampled, next_prefill_tokens,
                temperature, seeds, bt[g], self.swa_bss[i], self.block_tables.cp_rank, 1,
                self.block_tables.cp_interleave, self.mask_token_id, B, self.full,
                self.max_num_reqs, self.max_num_tokens, self.max_model_len, False)
        rows = (input_batch.query_start_loc[1:R + 1].to(torch.int64)
                - num_rejected[:R].to(torch.int64) - 1)
        self.prev_hidden[:R].copy_(last_hidden_states[rows])
        anchor_pos = self.input_buffers.positions[self._anchor_idx[:R]]
        for i, g in enumerate(self.swa_gids):
            _dfly_swa_kernel[(R,)](
                self.swa_idx[i], self.swa_len[i], self.corr_swa_idx, self.corr_swa_len,
                anchor_pos, bt[g], bt[g].stride(0), self.swa_bss[i],
                WINDOW=WINDOW, BLOCK=B, WS=SWA_WIDTH, WC=CORR_WIDTH, PW=256,
                WRITE_CORR=(i == 2))
        tk = [state.TOPK_BUFFERS[L] for L in self.target_layers]
        _dfly_topk_kernel[(R,)](
            self.topk_idx, self.topk_len, self.corr_topk_idx, self.corr_topk_len, rows,
            bt[self.comp_gid], bt[self.comp_gid].stride(0), self.comp_bs,
            tk[0], tk[1], tk[2], tk[0].stride(0), self.topk_idx.shape[1], BLOCK=B, TOPK=TOPK)
        self._static_ctx(R)
        dump = None
        self._step += 1
        if (self.dump_dir and self._dumped < self.dump_steps and R >= self.dump_min_reqs
                and self._step % self.dump_every == 0):
            dump = {"rows": rows.clone(), "anchor_pos": anchor_pos.clone()}
        Rp = next((s for s in self.graph_sizes if s >= R), None) if self.graphs else None
        if dump is None and Rp is not None and Rp in self.graphs:
            Tp = Rp * B
            if Tp > T:
                self.swa_len[:, T:Tp].zero_()
                self.topk_len[:, T:Tp].zero_()
                self.corr_swa_len[R:Rp].zero_()
                self.corr_topk_len[R:Rp].zero_()
            self.graphs[Rp][0].replay()
            return self.draft_tokens[:R]
        ctx.sched = [new_sched() for _ in range(3)]
        ctx.corr_sched = new_sched()
        ctx.dump = dump
        self._draft(R, sample=True)
        if dump is not None:
            self._write_dump(dump, R)
        ctx.dump = None
        return self.draft_tokens[:R]

    def _draft(self, R: int, sample: bool) -> None:
        B = self.block
        T = R * B
        lm = self.model
        model = lm.model
        ctx = model.step
        corr = model.correction
        attn = model.layers[-1].attn
        with set_forward_context(None, self.vllm_config, num_tokens=T,
                                 cudagraph_runtime_mode=CUDAGraphMode.NONE):
            input_ids = self.input_buffers.input_ids[:T]
            positions = self.input_buffers.positions[:T]
            hidden = lm(input_ids, positions)  # [T, D] after final_norm
            hidden = hidden.view(R, B, -1)
            pos = positions.view(R, B)
            anchor = input_ids.view(R, B)[:, 0]
            cos_sin = model.caches.cos_sin_cache
            # slot 0's key: previous token = the anchor itself
            prev = anchor
            keys = []
            cos0, sin0 = _rope_cos_sin(cos_sin, pos[:, 0])
            inp0 = corr.inputs(hidden[:, 0], model.embed_tokens(prev.contiguous()))
            keys.append(corr.block_key(inp0, cos0, sin0))
            dump = ctx.dump
            if dump is not None:
                dump["input_ids"] = input_ids.clone()
                dump["positions"] = positions.clone()
                dump["backbone"] = hidden.clone()
                dump["prev_hidden"] = ctx.prev_hidden[:R].clone()
            n = self.num_speculative_steps
            for j in range(1, n + 1):
                cos, sin = _rope_cos_sin(cos_sin, pos[:, j])
                inp = corr.inputs(hidden[:, j], model.embed_tokens(prev.contiguous()))
                keys.append(corr.block_key(inp, cos, sin))
                corrected = corr.forward_slot(attn, hidden[:, j], inp, keys, j, cos, sin, ctx)
                logits = lm.compute_logits(corrected)
                if dump is not None:
                    dump[f"logits{j}"] = logits.float().clone()
                    dump[f"corrected{j}"] = corrected.clone()
                if not sample:
                    prev = logits.argmax(-1)
                    continue
                idx_map = self.s_idx_map[: R * self.full].view(R, self.full)[:, j - 1]
                spos = self.s_pos[: R * self.full].view(R, self.full)[:, j - 1]
                if self.draft_top_p < 1.0 and self.draft_logits is not None:
                    logits = self._nucleus(logits, idx_map)
                tok = self._sample_logits(logits, idx_map, spos, j - 1)
                self.draft_tokens[:R, j - 1] = tok
                prev = tok
            if dump is not None:
                dump["draft_tokens"] = self.draft_tokens[:R].clone()

    def _nucleus(self, logits: torch.Tensor, idx_map: torch.Tensor) -> torch.Tensor:
        """Restrict the draft distribution to its top-p nucleus at the request temperature
        (among the top DFLY2_DRAFT_TOPK candidates). The masked logits are both what the draft
        samples from and what the verifier reads as q, so verification stays exact."""
        t = self.temperature[idx_map.clamp(min=0)].float().clamp(min=1e-4).unsqueeze(-1)
        x = logits.float() / t
        lse = torch.logsumexp(x, -1, keepdim=True)
        vals, _ = x.topk(self.draft_topk_cand, dim=-1)
        pr = torch.exp(vals - lse)
        before = torch.cumsum(pr, -1) - pr
        keep = before < self.draft_top_p
        kth = torch.where(keep, vals, torch.full_like(vals, float("inf"))).min(-1, keepdim=True).values
        return torch.where(x >= kth, logits, torch.full_like(logits, float("-inf")))

    def _write_dump(self, dump: dict, R: int) -> None:
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
        caches = self.model.model.caches
        T = R * self.block
        # Dequantized context latents the drafter attends to, for the oracle.
        lat = {}
        for i in range(3):
            swa = caches.swa_cache(i)
            comp = caches.compressed_cache(i)
            lat[f"window{i}"] = dequant_records(swa, self.swa_idx[i, :T:self.block],
                                                self.swa_bss[i])
            lat[f"topk{i}"] = dequant_records(comp, self.topk_idx[i, :T:self.block], self.comp_bs)
        dump.update(lat)
        dump["swa_len"] = self.swa_len[:, :T:self.block].clone()
        dump["topk_len"] = self.topk_len[:, :T:self.block].clone()
        dump["swa_idx"] = self.swa_idx[:, :T:self.block].clone()
        dump["topk_idx"] = self.topk_idx[:, :T:self.block].clone()
        os.makedirs(self.dump_dir, exist_ok=True)
        path = os.path.join(self.dump_dir, f"step{self._dumped:03d}_rank{rank}.pt")
        torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in dump.items()}, path)
        self._dumped += 1


def dequant_records(cache: torch.Tensor, slots: torch.Tensor, block_size: int) -> torch.Tensor:
    """bf16 latents [R, W, 512] of fp8_ds_mla V4 records (584 B: 448 e4m3 + 64 bf16 RoPE, then a
    page's scale rows of 7 e8m0 + pad) at ``slots`` (-1 -> zeros)."""
    raw = cache.view(torch.uint8)
    R, W = slots.shape
    s = slots.long().clamp(min=0)
    blk, off = s // block_size, s % block_size
    data = raw[blk.flatten()].reshape(R * W, -1)  # copies only the pages read
    off_f = off.flatten()
    row_bytes = 576
    idx = off_f[:, None] * row_bytes + torch.arange(row_bytes, device=cache.device)
    rows = torch.gather(data, 1, idx)
    sidx = block_size * row_bytes + off_f[:, None] * 8 + torch.arange(8, device=cache.device)
    sc = torch.gather(data, 1, sidx)[:, :7]
    nope = rows[:, :448].view(torch.float8_e4m3fn).float().view(-1, 7, 64)
    scale = torch.exp2(sc.float() - 127.0).view(-1, 7, 1)
    nope = (nope * scale).view(-1, 448)
    rope = rows[:, 448:576].contiguous().view(torch.bfloat16).float()
    out = torch.cat([nope, rope], -1).view(R, W, 512)
    return torch.where((slots >= 0)[..., None], out, torch.zeros_like(out)).to(torch.bfloat16)
