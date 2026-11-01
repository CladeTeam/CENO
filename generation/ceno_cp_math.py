# SPDX-License-Identifier: Apache-2.0
# NVIDIA TransformerEngine abbdd76 CP attention arithmetic, Triton adaptation.
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _lse_add(Total, Part, T: tl.constexpr, FULL: tl.constexpr,
             OFFSET: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    head=off//T;row=off%T;mask=off<H*T
    ptr=Total+head*FULL+row+OFFSET
    total=tl.load(ptr,mask=mask,other=0)
    part=tl.load(Part+off,mask=mask,other=0).to(tl.float64)
    hi=tl.maximum(total,part);lo=tl.minimum(total,part)
    value=hi+libdevice.log1p(libdevice.exp(lo-hi))
    tl.store(ptr,value,mask=mask)


@triton.jit
def _merge(Out,Part,Total,Lse,N:tl.constexpr,T:tl.constexpr,FULL:tl.constexpr,
           OFFSET:tl.constexpr,ADD:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
           BLOCK:tl.constexpr):
    off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=off//(H*D);head=(off//D)%H;mask=off<N
    total=tl.load(Total+head*FULL+row+OFFSET,mask=mask,other=0)
    lse=tl.load(Lse+head*T+row,mask=mask,other=0)
    # TE's Torch 2.7 compiled kernel uses approximate exp and fused add/multiply.
    # Current Torch's eager/compiled exp does not reproduce this BF16 rounding.
    weight=tl.exp(lse-total)
    value=tl.load(Part+off,mask=mask,other=0).to(tl.float32)
    dest=Out+OFFSET*H*D+off
    if ADD:
        old=tl.load(dest,mask=mask,other=0).to(tl.float32)
        result=tl.fma(value,weight,old)
    else:result=value*weight
    tl.store(dest,result,mask=mask)


def combine(parts, offsets):
    """Merge in ring order; every output update is stored back to BF16."""
    first,lse=parts[0];length,h,d=first.shape
    total=lse.double().clone()
    for (_,part),offset in zip(parts[1:],offsets[1:]):
        _lse_add[(triton.cdiv(part.numel(),256),)](
            total,part,part.shape[1],length,offset,h,256,enable_fp_fusion=False)
    total=total.float();out=torch.empty_like(first)
    for i,((value,part),offset) in enumerate(zip(parts,offsets)):
        _merge[(triton.cdiv(value.numel(),1024),)](
            out,value,total,part,value.numel(),len(value),length,offset,i>0,h,d,1024,
            enable_fp_fusion=False)
    return out
