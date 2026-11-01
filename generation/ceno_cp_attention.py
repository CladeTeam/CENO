# SPDX-License-Identifier: Apache-2.0
# CENO adaptation of NVIDIA TransformerEngine abbdd76 CP attention arithmetic.
"""Fixed-reference CP prefill with graph-safe single-token decode.

The reference length is explicit. It never depends on serving batch/prefix size.
"""
import torch
from vllm.v1.attention.backends import flash_attn as backend
from .ceno_cp_math import combine
from .ceno_attention import _gather

def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None, output_scale=None, output_block_scale=None):
    cp = self._ceno_reference_cp_size
    reference_length = self._ceno_reference_context_length
    if attn_metadata is not None and attn_metadata.max_query_len == 1:
        from .ceno_cp_decode import forward as decode
        b = attn_metadata.query_start_loc.numel() - 1
        return decode(query[:b], kv_cache, attn_metadata.block_table, attn_metadata.seq_lens[:b], reference_length, self._ceno_max_len, self.scale, output[:b], cp)
    b = self._ceno_max_seqs if attn_metadata is None else attn_metadata.query_start_loc.numel() - 1
    k = torch.empty((b * self._ceno_max_len, self.num_kv_heads, self.head_size), device=query.device, dtype=query.dtype)
    v = torch.empty_like(k)
    if attn_metadata is None:
        return output.zero_()
    m = attn_metadata
    (pk, pv) = kv_cache.unbind(0)
    lengths = m.seq_lens[:b]
    offsets = torch.zeros(b + 1, device=query.device, dtype=torch.int32)
    torch.cumsum(lengths, 0, out=offsets[1:])
    _gather[256, b](pk, pv, m.block_table, lengths, offsets, k, v, m.block_table.stride(0), pk.shape[1], self.num_kv_heads, self.head_size, *pk.stride(), 1024)
    lens = lengths.tolist()
    qstarts = m.query_start_loc.tolist()
    kstart = 0
    block = reference_length // (2 * cp)

    def ints(x):
        return torch.tensor(x, device=query.device, dtype=torch.int32)
    for (req, length) in enumerate(lens):
        (qs, qe) = qstarts[req:req + 2]
        q_begin = length - (qe - qs)
        (kk, vv) = (k[kstart:kstart + length], v[kstart:kstart + length])
        kstart += length

        def segment(x, chunk):
            return x[chunk * block:min((chunk + 1) * block, length)]

        def kv_part(x, rank, both):
            first = segment(x, rank)
            return torch.cat([first, segment(x, 2 * cp - rank - 1)], 0) if both else first
        for rank in range(cp):
            intervals = []
            for chunk in (rank, 2 * cp - rank - 1):
                start = max(q_begin, chunk * block)
                end = min(length, (chunk + 1) * block)
                intervals.append((max(start - q_begin, 0), max(end - q_begin, 0)) if end > start else (0, 0))
            ((a0, a1), (b0, b1)) = intervals
            nfirst = a1 - a0
            nsecond = b1 - b0
            if not nfirst + nsecond:
                continue
            qlocal = torch.cat([query[qs + a0:qs + a1], query[qs + b0:qs + b1]], 0)
            parts = []
            part_offsets = []
            for step in range(cp):
                if step > rank and (not nsecond):
                    continue
                src = (rank - step) % cp
                qq = qlocal if step <= rank else qlocal[nfirst:]
                (kpart, vpart) = (kv_part(kk, src, step == 0 or step > rank), kv_part(vv, src, step == 0 or step > rank))
                (out, lse) = backend.flash_attn_varlen_func(qq, kpart, vpart, len(qq), ints([0, len(qq)]), len(kpart), cu_seqlens_k=ints([0, len(kpart)]), causal=step == 0, fa_version=2, num_splits=1, softmax_scale=self.scale, return_softmax_lse=True)
                parts.append((out, lse))
                part_offsets.append(0 if step <= rank else nfirst)
            out = combine(parts, part_offsets)
            if nfirst:
                output[qs + a0:qs + a1].copy_(out[:nfirst])
            if nsecond:
                output[qs + b0:qs + b1].copy_(out[nfirst:])
    return output
