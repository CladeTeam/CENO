"""Loading or running a CENO HF model must not change the caller's torch GEMM flags.

Regression for the 2026-11 Wave7 slowdown: configure_backend() used to disable
TF32/reduced-precision reduction and force cuBLASLt for the whole process, which
made an unrelated FP32 head ~2.4x slower. The reference flags now apply only
inside the model's forward.
"""
import importlib
import importlib.machinery
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]  # ceno_model/


def _flags():
    m = torch.backends.cuda.matmul
    return (m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction,
            m.allow_tf32, torch.backends.cudnn.allow_tf32,
            str(torch.backends.cuda.preferred_blas_library()))


REFERENCE = (False, False, False, False, "_BlasBackend.Cublaslt")


@pytest.fixture
def user_flags():
    saved = _flags()
    m = torch.backends.cuda.matmul
    m.allow_bf16_reduced_precision_reduction = True
    m.allow_fp16_reduced_precision_reduction = True
    m.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.preferred_blas_library("cublas")
    yield _flags()
    (m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction,
     m.allow_tf32, torch.backends.cudnn.allow_tf32) = saved[:4]
    torch.backends.cuda.preferred_blas_library(
        "cublaslt" if saved[4].endswith("Cublaslt") else "cublas")


@pytest.fixture
def modeling(monkeypatch):
    # CPU images may lack mamba_ssm; the attention/MLP-only model below does not
    # need it. Stub it so the import path is the same with or without it.
    def stub(name, package):
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=package)
        return module

    if importlib.util.find_spec("mamba_ssm") is None:
        for name in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton"):
            monkeypatch.setitem(sys.modules, name, stub(name, True))
        leaf = stub("mamba_ssm.ops.triton.layernorm_gated", False)
        leaf.rmsnorm_fn = None
        monkeypatch.setitem(sys.modules, leaf.__name__, leaf)
    monkeypatch.syspath_prepend(str(ROOT))
    for name in ("ceno_hf.modeling_ceno", "ceno_hf.training_precision"):
        sys.modules.pop(name, None)
    return importlib.import_module("ceno_hf.modeling_ceno")


def test_configure_backend_leaves_torch_flags(user_flags, modeling):
    precision = importlib.import_module("ceno_hf.training_precision")
    precision.configure_backend()
    assert _flags() == user_flags


def test_backend_scope_restores_on_exit_error_and_nesting(user_flags, modeling):
    precision = importlib.import_module("ceno_hf.training_precision")
    with precision.backend_scope():
        assert _flags() == REFERENCE
        with precision.backend_scope():
            assert _flags() == REFERENCE
        assert _flags() == REFERENCE
    assert _flags() == user_flags
    with pytest.raises(RuntimeError):
        with precision.backend_scope():
            raise RuntimeError("boom")
    assert _flags() == user_flags


def test_from_pretrained_and_forward_keep_global_flags(user_flags, modeling):
    from ceno_hf.configuration_ceno import CENOConfig
    cfg = CENOConfig(vocab_size=512, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, hybrid_override_pattern="*-",
                          num_attention_heads=4, num_key_value_heads=4, attention_head_dim=16)
    cfg.ceno_training_precision = True
    cfg.tie_word_embeddings = False
    if isinstance(getattr(modeling.CENOForCausalLM, "_tied_weights_keys", None), list):
        # Newer transformers expects a mapping here; tying is irrelevant to this test.
        modeling.CENOForCausalLM._tied_weights_keys = None
    with tempfile.TemporaryDirectory() as path:
        modeling.CENOForCausalLM(cfg).save_pretrained(path)
        model = modeling.CENOForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32, attn_implementation="eager").eval()
    # eager keeps the original backend: no scope, flags untouched.
    assert not getattr(model, "_ceno_backend_scope", False)
    with tempfile.TemporaryDirectory() as path:
        modeling.CENOForCausalLM(cfg).save_pretrained(path)
        model = modeling.CENOForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32).eval()  # default backend (FA2 on CUDA)
    assert _flags() == user_flags, "loading changed process-wide GEMM flags"
    assert model._ceno_backend_scope and model.backbone._ceno_backend_scope

    if not torch.cuda.is_available():
        return  # the forward half runs in the CUDA validation job
    model = model.to("cuda", torch.bfloat16)
    # The tiny test architecture is outside the precision contract; skip the lazy
    # kernel install and test only the flag scoping around forward.
    model.ceno_precision_reference = {}
    seen = []
    model.backbone.layers[0].register_forward_pre_hook(lambda *_: seen.append(_flags()))
    ids = torch.tensor([[65, 67, 71, 84] * 4], device="cuda")
    with torch.no_grad():
        model(ids, use_cache=False)            # full model (lm_head inside the scope)
        model.backbone(input_ids=ids, use_cache=False)  # Wave7 uses the backbone only
    assert seen == [REFERENCE, REFERENCE]
    assert _flags() == user_flags, "forward leaked GEMM flags"


def test_forward_wrapper_scopes_marked_modules_only(user_flags, modeling):
    seen = []

    @modeling._with_backend_scope
    def forward(self, x):
        seen.append(_flags())
        if x == "raise":
            raise RuntimeError("boom")
        return x

    marked = types.SimpleNamespace(_ceno_backend_scope=True)
    plain = types.SimpleNamespace()
    assert forward(marked, 1) == 1 and forward(plain, 2) == 2
    with pytest.raises(RuntimeError):
        forward(marked, "raise")
    assert seen == [REFERENCE, user_flags, REFERENCE]
    assert _flags() == user_flags
    assert forward.__wrapped__.__name__ == "forward"  # generate() inspects the signature
