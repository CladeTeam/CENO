"""Segmented Mamba prefill must reproduce the single-pass forward (CUDA kernels).

Long sequences are split into chunk-aligned segments once batch * length * in_proj width
reaches 2**31, because the causal_conv1d and mamba_ssm kernels use 32-bit element offsets.
Segments carry the convolution inputs and the SSM state, so the result must match one pass.
The test forces small segments on a tiny model; it needs a CUDA GPU with mamba_ssm and
causal_conv1d and is skipped otherwise.
"""
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]  # ceno_model/

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("mamba_ssm") is None
    or importlib.util.find_spec("causal_conv1d") is None,
    reason="needs CUDA with mamba_ssm and causal_conv1d",
)


@pytest.fixture(scope="module")
def model():
    sys.path.insert(0, str(ROOT))
    modeling = importlib.import_module("ceno_hf.modeling_ceno")
    from ceno_hf.configuration_ceno import CENOConfig
    torch.manual_seed(0)
    cfg = CENOConfig(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=5,
                     hybrid_override_pattern="MEM*E", num_attention_heads=4, num_key_value_heads=4,
                     attention_head_dim=32, mamba_num_heads=8, mamba_head_dim=32, ssm_state_size=32,
                     n_groups=2, chunk_size=64, num_experts=4, moe_top_k=2, conv_kernel=4,
                     use_conv_bias=True, mamba_in_proj_layernorm=True, ceno_mamba_gate_in_ssm=True,
                     qkv_layernorm=True, use_mamba_kernels=True)
    cfg.ceno_training_precision = False
    cfg._attn_implementation = "sdpa"
    m = modeling.CENOForCausalLM(cfg).cuda().float().eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.1)
    return m, modeling


def _set_segment(model, value):
    for block in model.backbone.layers:
        if block.block_type == "mamba":
            if value is None:
                block.mixer.__dict__.pop("_ceno_prefill_segment", None)
            else:
                block.mixer._ceno_prefill_segment = value


def _ids(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.tensor([65, 67, 71, 84])[torch.randint(0, 4, (n,), generator=g)].cuda()


@pytest.mark.parametrize("segment", [64, 128, 192])
def test_segmented_matches_single_pass(model, segment):
    m, _ = model
    ids = torch.stack([_ids(1000, 1), _ids(1000, 2)])
    mask = torch.ones_like(ids)
    mask[1, :70] = 0  # left padding crosses a segment boundary for segment 64
    with torch.no_grad():
        _set_segment(m, None)
        ref = m(ids, attention_mask=mask, use_cache=False).logits
        _set_segment(m, segment)
        seg = m(ids, attention_mask=mask, use_cache=False).logits
        _set_segment(m, None)
    assert float((ref - seg)[0].abs().max()) < 1e-4
    assert float((ref - seg)[1, 70:].abs().max()) < 1e-4


def test_segmented_prefill_cache_state(model):
    m, modeling = model
    ids = _ids(777, 3)[None]
    states = {}
    with torch.no_grad():
        for seg in (None, 128):
            _set_segment(m, seg)
            cache = modeling.HybridMambaAttentionDynamicCache(m.config, 1, dtype=torch.float32, device="cuda")
            m(ids, cache_params=cache, use_cache=True, cache_position=torch.arange(ids.shape[1], device="cuda"))
            states[seg] = ([s.clone() for s in cache.ssm_states], [c.clone() for c in cache.conv_states])
        _set_segment(m, None)
    pairs = [(a, b) for a, b in zip(states[None][0] + states[None][1], states[128][0] + states[128][1]) if a.numel()]
    assert pairs  # mamba layers carry conv and SSM state
    for a, b in pairs:
        assert float((a - b).abs().max()) < 1e-4


def test_segment_length_rule(model):
    m, _ = model
    mixer = next(b.mixer for b in m.backbone.layers if b.block_type == "mamba")
    width = mixer.intermediate_size + mixer.conv_dim + mixer.num_heads
    below = (2**31 - 1) // width
    assert mixer._prefill_segment_length(1, below) is None
    seg = mixer._prefill_segment_length(1, below + 1)
    assert seg % mixer.chunk_size == 0 and seg * width < 2**31
    seg2 = mixer._prefill_segment_length(4, below)
    assert seg2 % mixer.chunk_size == 0 and 4 * seg2 * width < 2**31
