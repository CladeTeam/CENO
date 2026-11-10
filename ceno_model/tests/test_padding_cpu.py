"""Padding must not change the real tokens' outputs (FP32, CPU reference path).

The model has no position embeddings; Mamba layers zero pad positions before the
input projection and after the convolution, and attention masks pad keys. So in
FP32 a padded sequence's real positions must equal the unpadded run up to
rounding, for left or right padding, for batch size 1 or more, and regardless
of what token ids sit under the mask.

Regression: Mamba skipped the zeroing for batch size 1, so a single left-padded
sequence fed the pad tokens into its convolution and SSM state (max logit error
~1 on the tiny model below; the batch>=2 path was already correct).
"""
import importlib
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]  # ceno_model/
TOL = 1e-5
BASES = torch.tensor([65, 67, 71, 84])
PAD = 1


def _stub(name, package):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=package)
    return module


@pytest.fixture(params=["modeling_ceno", "modeling_ceno_p"])
def model_cls(request, monkeypatch):
    if importlib.util.find_spec("mamba_ssm") is None:
        for name in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton"):
            monkeypatch.setitem(sys.modules, name, _stub(name, True))
        leaf = _stub("mamba_ssm.ops.triton.layernorm_gated", False)
        leaf.rmsnorm_fn = None
        monkeypatch.setitem(sys.modules, leaf.__name__, leaf)
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.syspath_prepend(str(ROOT / "ceno_hf"))  # modeling_ceno_p imports configuration_ceno flat
    for name in ("ceno_hf.modeling_ceno", "ceno_hf.modeling_ceno_p", "ceno_hf.training_precision"):
        sys.modules.pop(name, None)
    modeling = importlib.import_module(f"ceno_hf.{request.param}")
    return getattr(modeling, "CENOPForCausalLM" if request.param.endswith("_p") else "CENOForCausalLM")


@pytest.fixture(params=["sdpa", "eager"])
def model(request, model_cls):
    from ceno_hf.configuration_ceno import CENOConfig
    torch.manual_seed(0)
    cfg = CENOConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=5,
                     hybrid_override_pattern="MEM*E", num_attention_heads=4, num_key_value_heads=4,
                     attention_head_dim=16, mamba_num_heads=8, mamba_head_dim=16, ssm_state_size=16,
                     n_groups=1, chunk_size=16, num_experts=4, moe_top_k=2, conv_kernel=4,
                     use_conv_bias=True, mamba_in_proj_layernorm=True, ceno_mamba_gate_in_ssm=True,
                     qkv_layernorm=True, use_mamba_kernels=False)
    cfg.ceno_training_precision = False
    cfg._attn_implementation = request.param
    m = model_cls(cfg).eval()
    with torch.no_grad():
        for p in m.parameters():  # non-trivial conv/dt biases so leaked pads would show
            p.normal_(0, 0.2)
    return m


def _seq(n, seed):
    g = torch.Generator().manual_seed(seed)
    return BASES[torch.randint(0, 4, (n,), generator=g)]


def _logits(m, ids, mask, row, start, n):
    with torch.no_grad():
        return m(ids, attention_mask=mask, use_cache=False).logits[row, start:start + n]


def _cases():
    s, o = _seq(50, 1), _seq(80, 2)
    pad = lambda n: torch.full((n,), PAD)
    junk = lambda n: _seq(n, 3)  # real bases under the mask: must still be invisible
    m = lambda *rows: torch.tensor(rows)
    yield "b1_left", torch.cat([pad(7), s])[None], m([0] * 7 + [1] * 50), 0, 7
    yield "b1_left_junk", torch.cat([junk(7), s])[None], m([0] * 7 + [1] * 50), 0, 7
    yield "b1_right", torch.cat([s, pad(7)])[None], m([1] * 50 + [0] * 7), 0, 0
    yield "b2_self_left", torch.stack([torch.cat([pad(30), s]), o]), m([0] * 30 + [1] * 50, [1] * 80), 0, 30
    yield "b2_self_left_junk", torch.stack([torch.cat([junk(30), s]), o]), m([0] * 30 + [1] * 50, [1] * 80), 0, 30
    yield "b2_self_right", torch.stack([torch.cat([s, pad(30)]), o]), m([1] * 50 + [0] * 30, [1] * 80), 0, 0
    yield "b2_other_left", torch.stack([s, torch.cat([pad(20), o[:30]])]), m([1] * 50, [0] * 20 + [1] * 30), 0, 0
    yield "b2_other_right", torch.stack([s, torch.cat([o[:30], pad(20)])]), m([1] * 50, [1] * 30 + [0] * 20), 0, 0


def test_padding_does_not_change_real_tokens(model):
    s = _seq(50, 1)
    ref = _logits(model, s[None], None, 0, 0, 50)
    errors = {name: float((_logits(model, ids, mask, row, start, 50) - ref).abs().max())
              for name, ids, mask, row, start in _cases()}
    assert all(e < TOL for e in errors.values()), errors


def test_all_ones_mask_is_bitwise_unpadded(model):
    ids = torch.stack([_seq(50, 1), _seq(50, 2)])
    assert torch.equal(_logits(model, ids, None, 0, 0, 50),
                       _logits(model, ids, torch.ones_like(ids), 0, 0, 50))


def test_right_padding_prefill_uses_causal_kernel(model):
    # Right padding (tokenizer default) needs no explicit mask under causal attention; left padding does.
    backbone = model.backbone
    x = torch.zeros(2, 6, 64)
    pos = torch.arange(6)
    right = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    left = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
    interior = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 1, 1]])
    assert backbone._update_causal_mask(right, x, pos) is None
    assert backbone._update_causal_mask(left, x, pos) is not None
    assert backbone._update_causal_mask(interior, x, pos) is not None
    assert backbone._update_causal_mask(right, x, pos, needs_explicit_mask=True) is not None


def test_interior_gap_batch_size_independent(model):
    # Masked runs inside a sequence (e.g. N gaps) must give the same real-token outputs at any batch size.
    s, o = _seq(50, 1), _seq(50, 2)
    ids = s.clone(); ids[20:30] = PAD
    mask = torch.ones(50, dtype=torch.long); mask[20:30] = 0
    one = _logits(model, ids[None], mask[None], 0, 0, 50)
    two = _logits(model, torch.stack([ids, o]), torch.stack([mask, torch.ones_like(mask)]), 0, 0, 50)
    keep = mask.bool()
    assert float((one[keep] - two[keep]).abs().max()) < TOL
