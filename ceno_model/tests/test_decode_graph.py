"""B=1 CUDA-graph decode in ceno_hf.training_precision.

CPU part: the process-level cuBLAS setting and when the graph path is (not) taken.
CUDA part (needs mamba_ssm, FA2 and CENO_HF_TEST_CHECKPOINT=<HF export dir>):
graph decode is bit-identical to the eager decode across chunk boundaries.
"""
import importlib
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
tp = importlib.import_module("ceno_hf.training_precision")


def test_configure_backend_leaves_cublas_workspace_config(monkeypatch):
    # CUBLAS_WORKSPACE_CONFIG costs ~30-40 us of host time per GEMM in torch 2.7 and
    # does not change the cuBLASLt reference GEMMs; only the Lt workspace limit does.
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("CUBLASLT_WORKSPACE_SIZE", raising=False)
    tp.configure_backend()
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    assert os.environ["CUBLASLT_WORKSPACE_SIZE"] == "1"


class _Tensor:
    def __init__(self, *shape):
        self.shape = shape

    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n


def _fake(pos=5, decoded=True):
    layers = [types.SimpleNamespace(block_type=t) for t in ("mamba", "attention", "moe")]
    model = types.SimpleNamespace(
        backbone=types.SimpleNamespace(layers=layers),
        config=types.SimpleNamespace(use_cache=True, output_hidden_states=False, output_attentions=False))
    cache = types.SimpleNamespace(_ceno_chunks={0: {"pos": pos, "decoded": decoded}},
                                  attention_layers=[1], key_cache=[None, _Tensor(1, 8, 10, 64), None])
    modeling = types.SimpleNamespace(_ceno_scan_parts=object())
    kwargs = dict(input_ids=_Tensor(1, 1), cache_params=cache, cache_position=_Tensor(1), use_cache=True)
    return model, modeling, kwargs


@pytest.mark.parametrize("change", [
    dict(env="0"), dict(input_ids=_Tensor(2, 1)), dict(input_ids=_Tensor(1, 4)),
    dict(inputs_embeds=_Tensor(1, 1, 8)), dict(labels=_Tensor(1, 1)), dict(output_hidden_states=True),
    dict(use_cache=False), dict(cache_params=None), dict(cache_position=None),
    dict(seq_idx=_Tensor(1, 1)), dict(pos=tp._CHUNK - 1), dict(decoded=False), dict(parts=None),
])
def test_graph_path_falls_back_to_eager(monkeypatch, change):
    change = dict(change)
    model, modeling, kwargs = _fake(pos=change.pop("pos", 5), decoded=change.pop("decoded", True))
    if "env" in change:
        monkeypatch.setenv("CENO_HF_DECODE_GRAPH", change.pop("env"))
    if "parts" in change:
        modeling._ceno_scan_parts = change.pop("parts")
    kwargs.update(change)

    def fail(*args, **kwargs):
        raise AssertionError("graph path must not be taken")

    monkeypatch.setattr(tp, "_DecodeGraph", fail)
    assert tp._decode_graph_step(model, modeling, kwargs) is None


def _cuda_ready():
    try:
        import torch
        importlib.import_module("mamba_ssm")
        importlib.import_module("flash_attn")
    except ImportError:
        return False
    return torch.cuda.is_available() and bool(os.environ.get("CENO_HF_TEST_CHECKPOINT"))


@pytest.mark.skipif(not _cuda_ready(), reason="needs CUDA, mamba_ssm, FA2 and CENO_HF_TEST_CHECKPOINT")
def test_graph_decode_bitwise_equals_eager(monkeypatch):
    import torch
    model = tp.load_model(os.environ["CENO_HF_TEST_CHECKPOINT"])
    modeling = sys.modules[type(model).__module__]
    g = torch.Generator().manual_seed(7)
    seq = torch.tensor([65, 67, 71, 84])[torch.randint(0, 4, (700,), generator=g)].tolist()
    prompt, steps = 300, 300  # crosses chunk boundaries (positions 383, 511) and position 0

    def decode(flag):
        monkeypatch.setenv("CENO_HF_DECODE_GRAPH", flag)
        cache = modeling.HybridMambaAttentionDynamicCache(model.config, 1, model.dtype, device=model.device)
        ids, rows = torch.tensor([seq[:prompt]], device="cuda"), []
        with torch.inference_mode():
            for step in range(steps):
                pos = (torch.arange(prompt, device="cuda") if step == 0
                       else torch.tensor([prompt + step - 1], device="cuda"))
                rows.append(model(ids, cache_params=cache, use_cache=True, cache_position=pos).logits[:, -1].cpu())
                ids = torch.tensor([[seq[prompt + step]]], device="cuda")
        return torch.cat(rows), hasattr(cache, "_ceno_decode_graph")

    eager, used_eager = decode("0")
    graph, used_graph = decode("1")
    assert not used_eager and used_graph
    assert torch.equal(eager, graph)
