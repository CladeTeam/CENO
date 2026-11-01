# SPDX-License-Identifier: Apache-2.0
# CENO inference; see third_party licenses and NOTICE for upstream attribution.
"""Batch independent virtual CP partials into one FA2 decode launch."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _active(length, STEP, CHUNK: tl.constexpr, CP: tl.constexpr):
    chunk = tl.maximum(length - 1, 0) // CHUNK
    rank = tl.minimum(chunk, 2 * CP - chunk - 1)
    return (length > 0) & ((STEP <= rank) | (chunk >= CP))

@triton.jit
def _lengths(Lengths, Part, First, B: tl.constexpr, CHUNK: tl.constexpr, CP: tl.constexpr, BLOCK: tl.constexpr):
    part = tl.arange(0, BLOCK)
    req = part % B
    step = part // B
    length = tl.load(Lengths + req, part < B * CP, other=0)
    active = _active(length, step, CHUNK, CP)
    chunk = tl.maximum(length - 1, 0) // CHUNK
    rank = tl.minimum(chunk, 2 * CP - chunk - 1)
    src = (rank - step + CP) % CP
    first = tl.maximum(0, tl.minimum(CHUNK, length - src * CHUNK))
    both = (step == 0) | (step > rank)
    second = tl.where(both, tl.maximum(0, tl.minimum(CHUNK, length - (2 * CP - src - 1) * CHUNK)), 0)
    tl.store(Part + part, tl.where(active, first + second, 0), part < B * CP)
    tl.store(First + part, first, part < B * CP)

@triton.jit
def _gather(K, V, Table, Lengths, First, Offsets, OutK, OutV, OriginalLengths, TABLE_STRIDE: tl.constexpr, PAGE: tl.constexpr, H: tl.constexpr, D: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr, B: tl.constexpr, CHUNK: tl.constexpr, CP: tl.constexpr, BLOCK: tl.constexpr):
    part = tl.program_id(1)
    req = part % B
    step = part // B
    length = tl.load(Lengths + part)
    first = tl.load(First + part)
    start = tl.load(Offsets + part).to(tl.int64)
    original = tl.load(OriginalLengths + req)
    chunk = tl.maximum(original - 1, 0) // CHUNK
    rank = tl.minimum(chunk, 2 * CP - chunk - 1)
    src = (rank - step + CP) % CP
    for at in range(tl.program_id(0) * BLOCK, length * H * D, tl.num_programs(0) * BLOCK):
        off = at + tl.arange(0, BLOCK)
        row = off // (H * D)
        col = off % (H * D)
        valid = row < length
        logical = tl.where(row < first, src * CHUNK + row, (2 * CP - src - 1) * CHUNK + row - first)
        slot = tl.load(Table + req * TABLE_STRIDE + logical // PAGE, mask=valid, other=0).to(tl.int64)
        address = slot * S0 + logical % PAGE * S1 + col // D * S2 + col % D * S3
        k = tl.load(K + address, mask=valid, other=0)
        v = tl.load(V + address, mask=valid, other=0)
        target = start * H * D + off
        tl.store(OutK + target, k, mask=valid)
        tl.store(OutV + target, v, mask=valid)

@triton.jit
def _lse_add(Total, Part, Lengths, B: tl.constexpr, H: tl.constexpr, STEP: tl.constexpr, CHUNK: tl.constexpr, CP: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    req = off % B
    length = tl.load(Lengths + req)
    mask = (off < H * B) & _active(length, STEP, CHUNK, CP)
    total = tl.load(Total + off, mask=mask, other=0)
    part = tl.load(Part + off, mask=mask, other=0).to(tl.float64)
    hi = tl.maximum(total, part)
    lo = tl.minimum(total, part)
    value = hi + libdevice.log1p(libdevice.exp(lo - hi))
    tl.store(Total + off, value, mask=mask)

@triton.jit
def _merge(Out, Part, Total, Lse, Lengths, B: tl.constexpr, H: tl.constexpr, D: tl.constexpr, STEP: tl.constexpr, ADD: tl.constexpr, CHUNK: tl.constexpr, CP: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    req = off // (H * D)
    head = off // D % H
    length = tl.load(Lengths + req, mask=req < B, other=0)
    mask = (req < B) & _active(length, STEP, CHUNK, CP)
    total = tl.load(Total + head * B + req, mask=mask, other=0)
    lse = tl.load(Lse + head * B + req, mask=mask, other=0)
    weight = tl.exp(lse - total)
    value = tl.load(Part + off, mask=mask, other=0).to(tl.float32)
    if ADD:
        old = tl.load(Out + off, mask=mask, other=0).to(tl.float32)
        result = tl.fma(value, weight, old)
    else:
        result = value * weight
    tl.store(Out + off, result, mask=mask)

def forward(query, cache, table, lengths, reference_length, max_model_len, scale, output=None, cp=8):
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    (b, h, d) = query.shape
    chunk = reference_length // (2 * cp)
    (pk, pv) = cache.unbind(0)
    output = torch.empty_like(query) if output is None else output
    output.zero_()
    size = b * cp
    part_lengths = torch.empty(size, device=query.device, dtype=torch.int32)
    first = torch.empty_like(part_lengths)
    _lengths[1,](lengths, part_lengths, first, b, chunk, cp, triton.next_power_of_2(size))
    cuk = torch.zeros(size + 1, device=query.device, dtype=torch.int32)
    torch.cumsum(part_lengths, 0, out=cuk[1:])
    k = torch.empty((b * max_model_len, h, d), device=query.device, dtype=query.dtype)
    v = torch.empty_like(k)
    _gather[256, size](pk, pv, table, part_lengths, first, cuk, k, v, lengths, table.stride(0), pk.shape[1], h, d, *pk.stride(), b, chunk, cp, 1024)
    cuq = torch.arange(size + 1, device=query.device, dtype=torch.int32)
    (values, lse) = flash_attn_varlen_func(query.repeat(cp, 1, 1), k, v, 1, cuq, min(2 * chunk, max_model_len), cu_seqlens_k=cuk, causal=False, softmax_scale=scale, fa_version=2, num_splits=1, return_softmax_lse=True)
    values = values.view(cp, b, h, d)
    parts = [(values[step], lse[:, step * b:(step + 1) * b].contiguous()) for step in range(cp)]
    total = parts[0][1].double().clone()
    for (step, (_, part_lse)) in enumerate(parts[1:], 1):
        _lse_add[triton.cdiv(h * b, 256),](total, part_lse, lengths, b, h, step, chunk, cp, 256, enable_fp_fusion=False)
    total = total.float()
    for (step, (part, part_lse)) in enumerate(parts):
        _merge[triton.cdiv(b * h * d, 1024),](output, part, total, part_lse, lengths, b, h, d, step, step > 0, chunk, cp, 1024, enable_fp_fusion=False)
    return output
