# SPDX-License-Identifier: Apache-2.0
# CENO inference; see third_party licenses and NOTICE for upstream attribution.
"""CENO ReLU2 MoE with Megatron's route scaling before the down projection.

The native vLLM GEMMs, expert packing and CUDA graph execution are retained.
Supports unquantized BF16, TP=EP=1, non-gated experts and top-k=2.
"""
import torch

@torch.library.custom_op('ceno::rms_norm', mutates_args=())
def ceno_rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Preserve the BF16 residual boundary through torch.compile/CUDA graphs."""
    value = x.float()
    normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon)
    return (normalized * weight.float()).to(x.dtype)

@ceno_rms_norm.register_fake
def _norm_fake(x, weight, epsilon):
    return torch.empty_like(x)
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _divide(a, b):
    return tl.inline_asm_elementwise('div.rn.f32 $0, $1, $2;', constraints='=f,f,f', args=[a, b], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def _routes(Selected, Ids, Weights, IntIds, B: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(Selected + row * 2, row < B, other=0)
    b = tl.load(Selected + row * 2 + 1, row < B, other=0)
    high = tl.maximum(a, b)
    da = a - high
    db = b - high
    ea = tl.where(da < -80.0, libdevice.exp(da.to(tl.float64)).to(tl.float32), libdevice.exp(da))
    eb = tl.where(db < -80.0, libdevice.exp(db.to(tl.float64)).to(tl.float32), libdevice.exp(db))
    denom = ea + eb
    tl.store(Weights + row * 2, _divide(ea, denom), row < B)
    tl.store(Weights + row * 2 + 1, _divide(eb, denom), row < B)
    ia = tl.load(Ids + row * 2, row < B, other=0).to(tl.int32)
    ib = tl.load(Ids + row * 2 + 1, row < B, other=0).to(tl.int32)
    tl.store(IntIds + row * 2, ia, row < B)
    tl.store(IntIds + row * 2 + 1, ib, row < B)

@triton.jit
def _activation_scale(Up, Weights, N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(Up + off, off < N, other=0).to(tl.float32)
    x = tl.maximum(x, 0.0)
    squared = (x * x).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(Weights + off // WIDTH, off < N, other=0).to(tl.float32)
    tl.store(Up + off, squared * weight, off < N)

@torch.library.custom_op('ceno::bf16_moe', mutates_args=())
def bf16_moe(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, router_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    import triton.language as tl
    from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel, try_get_optimal_moe_config
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
    if x.shape[0] == 0:
        return torch.empty_like(x)
    if x.shape[0] > 32768:
        return torch.cat([bf16_moe(x[i:i + 32768], w1, w2, router_logits[i:i + 32768], top_k) for i in range(0, x.shape[0], 32768)], dim=0)
    (experts, intermediate, hidden) = w1.shape
    if x.dtype != torch.bfloat16 or w2.shape != (experts, hidden, intermediate):
        raise ValueError('CENO MoE requires unquantized BF16 non-gated experts, TP=EP=1')
    if top_k != 2:
        raise ValueError('CENO expert reduction is validated for top-k=2 only')
    if vllm_is_batch_invariant():
        config = dict(BLOCK_SIZE_M=16 if x.shape[0] <= 8 else 64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=8, SPLIT_K=1, num_warps=4, num_stages=3)
    else:
        config = try_get_optimal_moe_config(w1.shape, w2.shape, top_k, None, x.shape[0])
    (selected, ids) = torch.topk(router_logits.float(), top_k, dim=-1)
    if vllm_is_batch_invariant():
        weights = torch.empty(selected.shape, dtype=x.dtype, device=x.device)
        int_ids = torch.empty(ids.shape, dtype=torch.int32, device=x.device)
        _routes[triton.cdiv(x.shape[0], 128),](selected, ids, weights, int_ids, x.shape[0], 128)
        ids = int_ids
    else:
        weights = torch.softmax(selected, dim=-1).to(x.dtype)
        ids = ids.to(torch.int32)
    (order, expert_ids, count) = moe_align_block_size(ids, config['BLOCK_SIZE_M'], experts)
    up = torch.empty((x.shape[0], top_k, intermediate), dtype=x.dtype, device=x.device)
    flags = dict(compute_type=tl.bfloat16, use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False)
    invoke_fused_moe_triton_kernel(x, w1, up, None, None, None, order, expert_ids, count, False, top_k, config, **flags)
    _activation_scale[triton.cdiv(up.numel(), 1024),](up, weights, up.numel(), intermediate, 1024)
    scaled = up
    down = torch.empty((x.shape[0], top_k, hidden), dtype=x.dtype, device=x.device)
    invoke_fused_moe_triton_kernel(scaled.view(-1, intermediate), w2, down, None, None, None, order, expert_ids, count, False, 1, config, **flags)
    return down[:, 0] + down[:, 1]

@bf16_moe.register_fake
def _fake(x, w1, w2, router_logits, top_k):
    return torch.empty_like(x)
