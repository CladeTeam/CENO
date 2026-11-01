# SPDX-License-Identifier: Apache-2.0
# CENO inference; see third_party licenses and NOTICE for upstream attribution.
"""Training-style Mamba chunk decode with state owned by vLLM's cache manager.

The original convolution cache and recurrent state remain in the first slices.
An open chunk and its initial state travel in the same cache pages, so request
slot reuse/preemption cannot leave an unrelated Python-side state behind.
"""
import torch
import triton
import triton.language as tl
CHUNK = 128

@triton.jit
def _reset(Cache, State, Slots, Lengths, CONV_SIZE: tl.constexpr, STATE_SIZE: tl.constexpr, CONV_STRIDE: tl.constexpr, STATE_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(1)
    slot = tl.load(Slots + b).to(tl.int64)
    length = tl.load(Lengths + b)
    if (slot >= 0) & (length == 1):
        for start in range(0, max(CONV_SIZE, STATE_SIZE), BLOCK):
            off = start + tl.arange(0, BLOCK)
            tl.store(Cache + slot * CONV_STRIDE + off, 0, mask=off < CONV_SIZE)
            tl.store(State + slot * STATE_STRIDE + off, 0, mask=off < STATE_SIZE)

def reset_new_decode(cache, slots, lengths):
    (conv, state) = cache
    if slots.numel():
        a = conv.shape[1] * conv.shape[2]
        b = state.shape[1] * state.shape[2] * state.shape[3]
        _reset[1, slots.numel()](conv, state, slots, lengths, a, b, conv.stride(0), state.stride(0), 1024)

def shapes(conv, ssm):
    """Reserve an open 128-token block, its cursor, and its initial SSM state."""
    (h, d, n) = ssm
    (conv_width, conv_dim) = conv
    inner = h * d
    r = 2 * inner + (conv_dim - inner) + h
    rows = triton.cdiv(CHUNK * r + 1, conv_dim)
    return ((conv_width + rows, conv_dim), (h, d, 2 * n))

@triton.jit
def _append(Cache, Slots, Current, Packed, Pos, B: tl.constexpr, R: tl.constexpr, CACHE_STRIDE: tl.constexpr, START: tl.constexpr, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(1)
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Slots + b).to(tl.int64)
    valid = slot >= 0
    pos = tl.load(Cache + slot * CACHE_STRIDE + COUNT, mask=valid, other=0).to(tl.int32)
    row = off // R
    col = off % R
    mask = (off < 128 * R) & valid
    old = tl.load(Cache + slot * CACHE_STRIDE + START + off, mask=mask & (row < pos), other=0)
    new = tl.load(Current + b * R + col, mask=mask & (row == pos), other=0)
    value = tl.where(row == pos, new, old)
    tl.store(Cache + slot * CACHE_STRIDE + START + off, new, mask=mask & (row == pos))
    tl.store(Packed + b * 128 * R + off, value, mask=off < 128 * R)
    if tl.program_id(0) == 0:
        tl.store(Pos + b, pos)

def _parts(packed, h, d, g, n):
    inner = h * d
    bn = g * n
    (x, B, C, dt, z) = packed.split([inner, bn, bn, h, inner], dim=-1)
    return (x.view(-1, h, d), dt.view(-1, h), B.view(-1, g, n), C.view(-1, g, n), z.view(-1, h, d))

def prefill(cache, slots, has_initial, conv_width, x, dt, A, B, C, **kw):
    """Rejoin any saved partial chunk before scanning newly scheduled tokens."""
    from vllm.model_executor.layers.mamba.ops.ssd_combined import mamba_chunk_scan_combined_varlen
    (conv, state) = cache
    (h, d) = x.shape[1:]
    (g, n) = B.shape[1:]
    inner = h * d
    if kw['chunk_size'] != CHUNK or slots.ndim != 1 or kw.get('return_intermediate_states'):
        raise ValueError('CENO chunk prefill requires 128-token chunks and no prefix caching')
    z = kw['z']
    if z is None:
        raise ValueError('CENO chunk prefill requires training gate placement')
    bounds = kw['cu_seqlens'].tolist()
    indices = slots.tolist()
    existing = has_initial.tolist() if has_initial is not None else [False] * len(indices)
    prefixes = []
    pieces = []
    initial = []
    lengths = [0]
    chunk_bounds = [0]
    chunk_ids = []
    last = []
    current = torch.cat([x.flatten(1), B.flatten(1), C.flatten(1), dt, z.flatten(1)], -1)
    r = current.shape[-1]
    for (seq, slot) in enumerate(indices):
        if slot < 0:
            raise ValueError('Unexpected padded prefill request')
        count = int(conv[slot, -1, -1].item()) if existing[seq] else 0
        if not 0 <= count < CHUNK:
            raise ValueError('Invalid saved chunk position')
        prefix = conv[slot, conv_width:, :].view(-1)[:count * r].view(count, r)
        piece = torch.cat([prefix, current[bounds[seq]:bounds[seq + 1]]], 0)
        pieces.append(piece)
        prefixes.append(count)
        initial.append(state[slot, ..., n:] if existing[seq] else torch.zeros_like(state[slot, ..., n:]))
        start = lengths[-1]
        end = start + len(piece)
        lengths.append(end)
        for at in range(start, end, CHUNK):
            chunk_ids.append(seq)
            chunk_bounds.append(min(at + CHUNK, end))
        last.append(len(chunk_ids) - 1)
    packed = torch.cat(pieces, 0)
    (xx, tt, bb, cc, zz) = _parts(packed, h, d, g, n)

    def ints(v):
        return torch.tensor(v, device=x.device, dtype=torch.int32)
    all_y = torch.empty_like(xx)
    base = torch.stack(initial).contiguous()
    states = mamba_chunk_scan_combined_varlen(xx, tt, A, bb, cc, chunk_size=CHUNK, cu_seqlens=ints(lengths), cu_chunk_seqlens=ints(chunk_bounds), last_chunk_indices=ints(last), seq_idx=ints(chunk_ids), out=all_y, D=kw['D'], z=zz, dt_bias=kw['dt_bias'], initial_states=base, dt_softplus=True, dt_limit=kw.get("dt_limit", (0.0, float("inf"))), state_dtype=torch.float32, return_intermediate_states=True)
    for (seq, slot) in enumerate(indices):
        (start, end) = lengths[seq:seq + 2]
        count = (end - start) % CHUNK
        li = last[seq]
        kw['out'][bounds[seq]:bounds[seq + 1]].copy_(all_y[start + prefixes[seq]:end])
        if count:
            first = 0 if seq == 0 else last[seq - 1] + 1
            before = base[seq] if li == first else states[li - 1]
            conv[slot, conv_width:, :].view(-1)[:count * r].copy_(packed[end - count:end].reshape(-1))
        else:
            before = states[li]
        state[slot, ..., n:].copy_(before)
        conv[slot, -1, -1] = count
    return states[ints(last).long()]

@triton.jit
def _row(Packed, State, Slots, Pos, DT, DA, Dptr, Out, H: tl.constexpr, D: tl.constexpr, G: tl.constexpr, N: tl.constexpr, B: tl.constexpr, R: tl.constexpr, STATE_STRIDE: tl.constexpr, M: tl.constexpr=16):
    req = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    slot = tl.load(Slots + req).to(tl.int64)
    pos = tl.load(Pos + req)
    group = head // (H // G)
    inner = H * D
    bn = G * N
    base = Packed + req * 128 * R
    mm = tl.arange(0, M)
    nn = tl.arange(0, 128)
    kk = tl.arange(0, 32)
    if slot >= 0:
        cb = tl.zeros((M, 128), tl.float32)
        for at in range(0, N, 32):
            c = tl.load(base + pos * R + inner + bn + group * N + at + kk[None, :] + tl.zeros((M, 1), tl.int32))
            b = tl.load(base + nn[None, :] * R + inner + group * N + at + kk[:, None])
            cb += tl.dot(c, b)
        da = tl.load(DA + head * B * 128 + req * 128 + nn)
        dt = tl.load(DT + head * B * 128 + req * 128 + nn)
        current = tl.load(DA + head * B * 128 + req * 128 + pos)
        cb *= tl.exp(current - da)[None, :]
        cb *= dt[None, :]
        cb = tl.where(nn[None, :] <= pos, cb, 0).to(Packed.dtype.element_ty)
        ks = tl.arange(0, 128)
        c = tl.load(base + pos * R + inner + bn + group * N + ks[None, :] + tl.zeros((M, 1), tl.int32))
        previous = tl.load(State + slot * STATE_STRIDE + (head * D + nn[None, :]) * 2 * N + N + ks[:, None], mask=nn[None, :] < D, other=0).to(Packed.dtype.element_ty)
        acc = tl.dot(c, previous) * tl.exp(current)
        cb = cb.reshape((M, 4, 32))
        for at in range(0, (pos // 64 + 1) * 64, 32):
            selected = tl.gather(cb, tl.full((M, 1, 32), at // 32, tl.int32), 1).reshape((M, 32))
            x = tl.load(base + (at + kk[:, None]) * R + head * D + nn[None, :], mask=nn[None, :] < D, other=0)
            acc += tl.dot(selected, x)
        residual = tl.load(base + pos * R + head * D + nn, nn < D, other=0).to(tl.float32)
        d = tl.load(Dptr + head).to(tl.float32)
        acc += residual[None, :] * d
        z = tl.load(base + pos * R + inner + 2 * bn + H + head * D + nn, nn < D, other=0).to(tl.float32)
        acc *= z[None, :] * tl.sigmoid(z[None, :])
        tl.store(Out + req * H * D + head * D + nn[None, :] + tl.zeros((M, 1), tl.int32), acc, mask=(mm[:, None] == 0) & (nn[None, :] < D))
    else:
        tl.store(Out + req * H * D + head * D + nn, 0, nn < D)

@triton.jit
def _boundary(Packed, State, Cache, Slots, Pos, DT, DA, H: tl.constexpr, D: tl.constexpr, G: tl.constexpr, N: tl.constexpr, B: tl.constexpr, R: tl.constexpr, STATE_STRIDE: tl.constexpr, CONV_STRIDE: tl.constexpr, COUNT: tl.constexpr):
    req = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    slot = tl.load(Slots + req).to(tl.int64)
    pos = tl.load(Pos + req)
    if (slot >= 0) & (pos == 127):
        base = Packed + req * 128 * R
        group = head // (H // G)
        mm = tl.arange(0, 64)
        nn = tl.arange(0, 128)
        kk = tl.arange(0, 32)
        last = tl.load(DA + head * B * 128 + req * 128 + 127)
        acc = tl.zeros((64, 128), tl.float32)
        for at in range(0, 128, 32):
            x = tl.load(base + (at + kk[None, :]) * R + head * D + mm[:, None])
            b = tl.load(base + (at + kk[:, None]) * R + H * D + group * N + nn[None, :]).to(tl.float32)
            da = tl.load(DA + head * B * 128 + req * 128 + at + kk)
            dt = tl.load(DT + head * B * 128 + req * 128 + at + kk)
            scale = tl.exp(last - da) * dt
            b *= scale[:, None]
            b = b.to(Packed.dtype.element_ty)
            acc += tl.dot(x, b)
        pointer = State + slot * STATE_STRIDE + (head * D + mm[:, None]) * 2 * N + N + nn[None, :]
        initial = tl.load(pointer)
        final = tl.fma(tl.exp(last), initial, acc)
        tl.store(pointer, final)
    if head == 0:
        tl.store(Cache + slot * CONV_STRIDE + COUNT, (pos + 1) % 128, slot >= 0)

def decode(cache, slots, x, dt, A, B, C, D, z, dt_bias, out, conv_width, dt_limit=(0.0, float("inf"))):
    from vllm.model_executor.layers.mamba.ops.ssd_chunk_state import _chunk_cumsum_fwd
    (conv, state) = cache
    (batch, heads, dim) = x.shape
    (groups, n) = B.shape[1:]
    assert dim == 64 and n == 128 and (z is not None)
    current = torch.cat([x.flatten(1), B.flatten(1), C.flatten(1), dt[:, :, 0], z.flatten(1)], -1)
    r = current.shape[-1]
    packed = torch.empty((batch * CHUNK, r), device=x.device, dtype=x.dtype)
    pos = torch.empty(batch, device=x.device, dtype=torch.long)
    count = conv.shape[1] * conv.shape[2] - 1
    _append[triton.cdiv(CHUNK * r, 1024), batch](conv, slots, current, packed, pos, batch, r, conv.stride(0), conv_width * conv.shape[-1], count, 1024)
    (_, raw_dt, _, _, _) = _parts(packed, heads, dim, groups, n)
    bounds = torch.arange(batch + 1, device=x.device, dtype=torch.int32) * CHUNK
    (da, processed_dt) = _chunk_cumsum_fwd(raw_dt, A, CHUNK, bounds, dt_bias=dt_bias, dt_softplus=True, dt_limit=dt_limit)
    _row[batch, heads](packed, state, slots, pos, processed_dt, da, D, out, heads, dim, groups, n, batch, r, state.stride(0), num_warps=4, num_stages=3)
    _boundary[batch, heads](packed, state, conv, slots, pos, processed_dt, da, heads, dim, groups, n, batch, r, state.stride(0), conv.stride(0), count, num_warps=4, num_stages=3)
