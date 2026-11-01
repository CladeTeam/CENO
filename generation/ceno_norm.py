# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# CENO Triton adaptation of TransformerEngine abbdd76 RMSNorm reduction order.
"""Explicit, batch-independent TE RMSNorm arithmetic for BF16 CENO widths 512/1024.

The training TE build chooses two reduction algorithms according to row count.
Keep that choice explicit: ``te-general`` reproduces the original 5385-row
reference; ``te-tuned`` reproduces the aligned training-shaped kernel. Never
switch based on a serving batch's size. This is not a generic TE replacement.

Source: NVIDIA/TransformerEngine@abbdd76, common/normalization/rmsnorm.
See third_party/TransformerEngine-LICENSE for the upstream Apache-2.0 license.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _xor(value, offset: tl.constexpr):
    # One warp owns one row. Changing num_warps/layout changes this reduction.
    return tl.inline_asm_elementwise(
        "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;",
        constraints="=f,f,r",
        args=[value, offset],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _te_norm(X, W, Y, EPS: tl.constexpr, TUNED: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    lanes = tl.arange(0, 32)
    mean = tl.full((32,), 0, tl.float32)
    if TUNED:
        for i in tl.static_range(WIDTH // 32):
            cols = lanes * 8 + (i // 8) * 256 + i % 8
            mean = mean + tl.load(X + row * WIDTH + cols).to(tl.float32)
        for offset in tl.static_range(5):
            mean = mean + _xor(mean, 1 << offset)
        mean = mean * (1.0 / WIDTH)
    variance = tl.full((32,), 0, tl.float32)
    for i in tl.static_range(WIDTH // 32):
        cols = lanes * 8 + (i // 8) * 256 + i % 8
        x = tl.load(X + row * WIDTH + cols).to(tl.float32)
        delta = x - mean if TUNED else x
        variance = tl.fma(delta, delta, variance)
    for offset in tl.static_range(5):
        variance = variance + _xor(variance, 1 << offset)
    moment = variance * (1.0 / WIDTH)
    if TUNED:
        moment = tl.fma(mean, mean, moment)
    scale = tl.rsqrt(moment + EPS)
    for i in tl.static_range(WIDTH // 32):
        cols = lanes * 8 + (i // 8) * 256 + i % 8
        x = tl.load(X + row * WIDTH + cols).to(tl.float32)
        weight = tl.load(W + cols).to(tl.float32)
        tl.store(Y + row * WIDTH + cols, (scale * x) * weight)


def _te_rms_norm_impl(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float, mode: str
) -> torch.Tensor:
    if mode not in ("te-general", "te-tuned"):
        raise ValueError(f"Unsupported TE RMSNorm mode: {mode}")
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.shape[-1] not in (512, 1024)
        or weight.dtype != x.dtype
        or weight.shape != (x.shape[-1],)
        or weight.device != x.device
        or epsilon <= 0
    ):
        raise ValueError("TE RMSNorm requires CUDA BF16 width 512/1024 and a positive epsilon")
    x, weight = x.contiguous(), weight.contiguous()
    out = torch.empty_like(x)
    rows = x.numel() // x.shape[-1]
    if rows:
        _te_norm[(rows,)](x, weight, out, epsilon, mode == "te-tuned", x.shape[-1], num_warps=1)
    return out


def _te_rms_norm_fake(x, weight, epsilon, mode):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


# Multiple HF Hub repositories can import this file under different dynamic names.
# Register the shared operator once per process.
if hasattr(torch.ops.ceno, "te_rms_norm"):
    te_rms_norm = torch.ops.ceno.te_rms_norm.default
else:
    te_rms_norm = torch.library.custom_op("ceno::te_rms_norm", mutates_args=())(
        _te_rms_norm_impl
    )
    te_rms_norm.register_fake(_te_rms_norm_fake)
