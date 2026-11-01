"""Precision path: vLLM-owned paged cache, ordinary FA2 arithmetic.

Repacking K/V costs bandwidth and staging memory; no promise of serving speed.
All cache lifetime/eviction remains with vLLM. No Python request state is kept.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gather(
    K, V, Table, Lengths, Offsets, OutK, OutV,
    TABLE_STRIDE: tl.constexpr, PAGE: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(1)
    pid = tl.program_id(0)
    length = tl.load(Lengths + b)
    start = tl.load(Offsets + b).to(tl.int64)
    for at in range(pid * BLOCK, length * H * D, tl.num_programs(0) * BLOCK):
        off = at + tl.arange(0, BLOCK)
        row = off // (H * D)
        col = off % (H * D)
        valid = row < length
        slot = tl.load(Table + b * TABLE_STRIDE + row // PAGE, mask=valid, other=0).to(tl.int64)
        address = slot * S0 + row % PAGE * S1 + col // D * S2 + col % D * S3
        k = tl.load(K + address, mask=valid, other=0)
        v = tl.load(V + address, mask=valid, other=0)
        target = start * H * D + off
        tl.store(OutK + target, k, mask=valid)
        tl.store(OutV + target, v, mask=valid)


def install():
    from vllm.config import get_current_vllm_config
    from vllm.v1.attention.backends import flash_attn as backend
    original_init = backend.FlashAttentionImpl.__init__

    def constructor(self, *a, **kw):
        original_init(self, *a, **kw)
        c = get_current_vllm_config()
        if c.parallel_config.tensor_parallel_size != 1 or c.speculative_config:
            raise ValueError('Training precision requires TP=1 and no speculation')
        self._ceno_max_len = c.model_config.max_model_len
        self._ceno_max_seqs = c.scheduler_config.max_num_seqs
        hf = c.model_config.hf_config
        self._ceno_reference_cp_size = getattr(hf, 'ceno_reference_cp_size', 1)
        self._ceno_reference_context_length = getattr(hf, 'ceno_reference_context_length', None)
        if (self.kv_cache_dtype != 'auto' or self.sliding_window != (-1, -1)
                or self.alibi_slopes is not None):
            raise ValueError('Training precision requires unquantized full causal attention')

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None,
                output_scale=None, output_block_scale=None):
        if query.dtype != torch.bfloat16 or output_scale is not None or output_block_scale is not None:
            raise ValueError('Precision attention requires BF16 without output quantization')
        # The first CP chunk has only one causal partial. Its result is ordinary FA2.
        if (self._ceno_reference_cp_size == 8
                and self._ceno_max_len > self._ceno_reference_context_length // 16):
            from .ceno_cp_attention import forward as cp_forward
            return cp_forward(self, layer, query, key, value, kv_cache, attn_metadata,
                              output, output_scale, output_block_scale)
        b = self._ceno_max_seqs if attn_metadata is None else attn_metadata.query_start_loc.numel() - 1
        # Include K/V staging in vLLM's peak-memory profiling before cache allocation.
        capacity = b * self._ceno_max_len
        k = torch.empty((capacity, self.num_kv_heads, self.head_size),
                        device=query.device, dtype=query.dtype)
        v = torch.empty_like(k)
        if attn_metadata is None:
            return output.zero_()
        m = attn_metadata
        if m.use_cascade or not m.causal:
            raise ValueError('Precision attention requires ordinary causal requests')
        (pk, pv) = kv_cache.unbind(0)
        lengths = m.seq_lens[:b]
        offsets = torch.zeros(b + 1, device=query.device, dtype=torch.int32)
        torch.cumsum(lengths, 0, out=offsets[1:])
        _gather[256, b](pk, pv, m.block_table, lengths, offsets, k, v,
                       m.block_table.stride(0), pk.shape[1], self.num_kv_heads,
                       self.head_size, *pk.stride(), 1024)
        backend.flash_attn_varlen_func(
            q=query[:m.num_actual_tokens], k=k, v=v, out=output[:m.num_actual_tokens],
            max_seqlen_q=m.max_query_len, cu_seqlens_q=m.query_start_loc,
            max_seqlen_k=self._ceno_max_len, cu_seqlens_k=offsets,
            causal=True, softmax_scale=self.scale, fa_version=2, num_splits=1,
        )
        return output
    backend.FlashAttentionImpl.__init__ = constructor
    backend.FlashAttentionImpl.forward = forward
