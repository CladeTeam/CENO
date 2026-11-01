"""Opt-in HF inference against the same BF16 Megatron reference as vLLM.

Use from the repository with the existing training image (Mamba 2.2.4 / FA2).
Defaults in exported checkpoints are unchanged. This adapter is inference-only,
single-GPU, BF16, unpadded uniform-length batches. Its prefill token budget is
bounded to avoid the old Mamba kernels' large-stride indexing paths.
"""
import contextvars
import functools
import os
import types

_CHUNK = 128


def _local_module(name):
    import importlib
    return importlib.import_module(f"{__package__}.{name}" if __package__ else name)


def configure_backend():
    """Call before CUDA matrix multiplication; settings apply to this process."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUBLASLT_WORKSPACE_SIZE"] = "1"
    import torch
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.version.cuda is not None:
        torch.backends.cuda.preferred_blas_library("cublaslt")


def _pin_scan():
    from mamba_ssm.ops.triton import ssd_bmm, ssd_chunk_scan, ssd_chunk_state, ssd_state_passing
    choices = [
        (ssd_chunk_state, "_chunk_cumsum_fwd_kernel", {"BLOCK_SIZE_H": 4}, 4, 2),
        (ssd_chunk_state, "_chunk_state_fwd_kernel",
         {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, 4, 4),
        (ssd_chunk_scan, "_chunk_scan_fwd_kernel",
         {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, 4, 4),
        (ssd_bmm, "_bmm_chunk_fwd_kernel",
         {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, 2, 4),
        (ssd_state_passing, "_state_passing_fwd_kernel", {"BLOCK_SIZE": 256}, 4, 2),
    ]
    for module, name, wanted, warps, stages in choices:
        kernel = getattr(module, name)
        matches = [c for c in kernel.configs if c.kwargs == wanted
                   and c.num_warps == warps and c.num_stages == stages]
        if len(matches) != 1:
            raise RuntimeError(f"Unsupported HF Mamba kernel profile: {name}")
        kernel.configs[:] = matches
        kernel.cache.clear()


def _cache_buffers(cache):
    if not hasattr(cache, "_ceno_chunks"):
        cache._ceno_chunks = {}
        original = cache.reorder_cache

        def reorder(beam_idx):
            result = original(beam_idx)
            for state in cache._ceno_chunks.values():
                for key in ("initial", "x", "dt", "B", "C", "z"):
                    value = state[key]
                    state[key] = value.index_select(0, beam_idx.to(value.device))
            return result

        cache.reorder_cache = reorder
    return cache._ceno_chunks


def _install_scan_hooks(module):
    """Namespace hooks are inactive unless a precision-enabled mixer sets context."""
    import torch
    if hasattr(module, "_ceno_scan_context"):
        return module._ceno_scan_context
    context = contextvars.ContextVar("ceno_hf_mixer", default=None)
    scan, update = module.mamba_chunk_scan_combined, module.selective_state_update

    def scan_with_cache(x, dt, A, B, C, *args, **kwargs):
        result = scan(x, dt, A, B, C, *args, **kwargs)
        active = context.get()
        if active is None or active[1] is None:
            return result
        mixer, cache = active
        z = kwargs.get("z")
        if z is None or result[1].dtype != torch.float32:
            raise ValueError("HF precision needs the training gate and FP32 final state")
        batch, length, heads, dim = x.shape
        groups, state_dim = B.shape[2:]
        count = length % _CHUNK
        complete = length - count
        if count == 0:
            initial = result[1].clone()
        elif complete:
            prefix_kwargs = dict(kwargs, z=z[:, :complete].contiguous())
            initial = scan(x[:, :complete].contiguous(), dt[:, :complete].contiguous(),
                           A, B[:, :complete].contiguous(), C[:, :complete].contiguous(),
                           *args, **prefix_kwargs)[1]
        else:
            initial = torch.zeros_like(result[1])
        state = {"initial": initial, "pos": count}
        values = {"x": x, "dt": dt, "B": B, "C": C, "z": z}
        for key, value in values.items():
            buffer = torch.zeros((batch, _CHUNK, *value.shape[2:]),
                                 device=x.device, dtype=value.dtype)
            if count:
                buffer[:, :count].copy_(value[:, -count:])
            state[key] = buffer
        _cache_buffers(cache)[mixer.layer_idx] = state
        return result

    def update_from_chunk(state_tensor, x, dt, A, B, C, D, *args, **kwargs):
        active = context.get()
        if active is None:
            return update(state_tensor, x, dt, A, B, C, D, *args, **kwargs)
        mixer, cache = active
        state = _cache_buffers(cache)[mixer.layer_idx]
        pos = state["pos"]
        update(state_tensor, x, dt, A, B, C, D, *args, **kwargs)
        for key, value in (("x", x), ("dt", dt[:, :, 0]), ("B", B), ("C", C),
                           ("z", kwargs["z"])):
            if pos == 0:
                state[key].zero_()
            state[key][:, pos].copy_(value)
        y, final = scan(state["x"], state["dt"], -torch.exp(mixer.A_log.float()),
                        state["B"], state["C"], chunk_size=_CHUNK, D=mixer.D,
                        z=state["z"], dt_bias=mixer.dt_bias, dt_softplus=True,
                        dt_limit=mixer.time_step_limit, initial_states=state["initial"],
                        return_final_states=True)
        if pos == _CHUNK - 1:
            state["initial"].copy_(final)
            state_tensor.copy_(final)
        state["pos"] = (pos + 1) % _CHUNK
        return y[:, pos]

    module.mamba_chunk_scan_combined = scan_with_cache
    module.selective_state_update = update_from_chunk
    module._ceno_scan_context = context
    return context


def _attention(q, k, v, causal):
    """Ordinary FA2 always; dense FA2 can change arithmetic with query length."""
    import torch
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    cuq = torch.tensor([0, len(q)], device=q.device, dtype=torch.int32)
    cuk = torch.tensor([0, len(k)], device=q.device, dtype=torch.int32)
    result = _flash_attn_varlen_forward(
        q, k, v, cuq, cuk, len(q), len(k), dropout_p=0., softmax_scale=q.shape[-1] ** -.5,
        causal=causal, window_size_left=-1, window_size_right=0 if causal else -1,
        softcap=0., alibi_slopes=None, return_softmax=False, block_table=None,
    )
    return result[0], result[1]


def _cp_attention(q, k, v, reference_length):
    import torch
    combine = _local_module("ceno_cp_math").combine
    cp = 8
    block = reference_length // (2 * cp)
    length, begin = len(k), len(k) - len(q)
    out = torch.empty_like(q)

    def segment(x, chunk):
        return x[chunk * block:min((chunk + 1) * block, length)]

    def part(x, rank, both):
        first = segment(x, rank)
        return torch.cat([first, segment(x, 2 * cp - rank - 1)]) if both else first

    for rank in range(cp):
        intervals = []
        for chunk in (rank, 2 * cp - rank - 1):
            start, end = max(begin, chunk * block), min(length, (chunk + 1) * block)
            intervals.append((start - begin, end - begin) if end > start else (0, 0))
        (a0, a1), (b0, b1) = intervals
        first, second = a1 - a0, b1 - b0
        if not first + second:
            continue
        query = torch.cat([q[a0:a1], q[b0:b1]])
        pieces, offsets = [], []
        for step in range(cp):
            if step > rank and not second:
                continue
            src, both = (rank - step) % cp, step == 0 or step > rank
            query_part = query if step <= rank else query[first:]
            pieces.append(_attention(query_part, part(k, src, both), part(v, src, both), step == 0))
            offsets.append(0 if step <= rank else first)
        result = combine(pieces, offsets)
        if first:
            out[a0:a1].copy_(result[:first])
        if second:
            out[b0:b1].copy_(result[first:])
    return out


def enable_training_precision(model, *, norm_mode=None, reference_cp_size=None,
                              reference_context_length=None):
    """Attach the common inference implementation to an already loaded HF model."""
    if hasattr(model, "ceno_precision_reference"):
        return model
    configure_backend()
    import sys
    import torch
    te_rms_norm = _local_module("ceno_norm").te_rms_norm

    config = model.config
    modeling = sys.modules[type(model).__module__]
    prefix = type(model).__name__.removesuffix("ForCausalLM")
    norm_mode = norm_mode or getattr(config, "ceno_rmsnorm_mode", "te-tuned")
    reference_cp_size = (getattr(config, "ceno_reference_cp_size", 1)
                         if reference_cp_size is None else reference_cp_size)
    reference_context_length = (getattr(config, "ceno_reference_context_length", None)
                                if reference_context_length is None else reference_context_length)
    expected = dict(hidden_size=(512, 1024), mamba_num_heads=(16, 32), mamba_head_dim=(64,),
                    n_groups=(8,), ssm_state_size=(128,), chunk_size=(128,),
                    num_attention_heads=(8, 16), num_experts=(8,), moe_top_k=(2,))
    if (prefix not in ("NemotronH", "CENO")
            or any(getattr(config, key, None) not in values for key, values in expected.items())
            or config.num_key_value_heads != config.num_attention_heads):
        raise ValueError("HF training precision supports ungrouped CENO 80M/300M/600M/1B")
    if (getattr(model, "is_quantized", False)
            or getattr(config, "moe_router_pre_softmax", False)
            or getattr(config, "mlp_use_swiglu", False)
            or getattr(config, "mlp_hidden_act", "relu2") not in ("relu2", "squared_relu")
            or tuple(getattr(config, "time_step_limit", (0., float("inf")))) not in ((0., float("inf")), (0., 1e30))):
        raise ValueError("HF precision requires unquantized post-topk ReLU2 and default Mamba dt limits")
    if norm_mode not in ("te-general", "te-tuned") or reference_cp_size not in (1, 8):
        raise ValueError("Choose an explicit norm and CP reference")
    if reference_cp_size == 8 and (
        norm_mode != "te-tuned" or not isinstance(reference_context_length, int)
        or reference_context_length <= 0 or reference_context_length % 1024
    ):
        raise ValueError("CP8 requires te-tuned norm and a fixed reference window divisible by 1024")
    if model.device.type != "cuda" or model.dtype != torch.bfloat16 or not modeling.is_fast_path_available:
        raise ValueError("HF precision requires single-GPU BF16 and the CUDA Mamba/FA2 environment")
    if any(p.device != model.device for p in model.parameters()):
        raise ValueError("HF training precision requires one CUDA device")
    config.ceno_mamba_gate_in_ssm = True
    config.ceno_rmsnorm_mode = norm_mode
    config.ceno_reference_cp_size = reference_cp_size
    config.ceno_reference_context_length = reference_context_length
    _pin_scan()
    context = _install_scan_hooks(modeling)
    def norm(self, x):
        return te_rms_norm(x, self.weight, self.variance_epsilon, norm_mode)

    def softmax(x, dim=-1):
        exp = torch.exp(x - torch.amax(x, dim=dim, keepdim=True))
        return exp / torch.sum(exp, dim=dim, keepdim=True)

    def attention(q, k, v, mask, query_length, **kwargs):
        if kwargs.get("sliding_window") is not None or kwargs.get("dropout", 0) != 0:
            raise ValueError("Only full causal inference attention is validated")
        if reference_cp_size == 8 and k.shape[1] > reference_context_length:
            raise ValueError("Sequence exceeds the fixed CP reference window")
        results = []
        for batch in range(q.shape[0]):
            if reference_cp_size == 1 or k.shape[1] <= reference_context_length // 16:
                results.append(_attention(q[batch], k[batch], v[batch], True)[0])
            else:
                results.append(_cp_attention(q[batch], k[batch], v[batch], reference_context_length))
        return torch.stack(results)

    for child in model.modules():
        if isinstance(child, getattr(modeling, prefix + "RMSNorm")):
            child.forward = types.MethodType(norm, child)
        elif isinstance(child, getattr(modeling, prefix + "MoE")):
            child._ceno_router_softmax = softmax
        elif isinstance(child, getattr(modeling, prefix + "Attention")):
            child.__class__ = getattr(modeling, prefix + "FlashAttention2")
            child._flash_attn_uses_top_left_mask = False
            child._ceno_attention = attention
        elif isinstance(child, getattr(modeling, prefix + "Mamba2Mixer")):
            original = child.cuda_kernels_forward

            def mixer_forward(self, hidden_states, cache_params=None, *args,
                              _original=original, **kwargs):
                token = context.set((self, cache_params))
                try:
                    return _original(hidden_states, cache_params, *args, **kwargs)
                finally:
                    context.reset(token)

            child.cuda_kernels_forward = types.MethodType(mixer_forward, child)

    original_forward = model.forward

    @functools.wraps(original_forward)
    def checked_forward(*args, **kwargs):
        if model.training or torch.is_grad_enabled():
            raise ValueError("Training precision is inference-only; use eval and no_grad/inference_mode")
        if kwargs.get("output_attentions") or model.config.output_attentions:
            raise ValueError("Reload with attn_implementation='eager' to inspect attention maps")
        ids = args[0] if args else kwargs.get("input_ids")
        inputs = ids if ids is not None else kwargs.get("inputs_embeds")
        if inputs is not None and inputs.shape[0] * inputs.shape[1] > 131072:
            raise ValueError("HF precision prefill budget is 131072 tokens; use vLLM for larger prefill")
        if kwargs.get("seq_idx") is not None:
            raise ValueError("Base generation precision does not implement MSA segments")
        mask = kwargs.get("attention_mask")
        if mask is not None and not bool(torch.all(mask)):
            raise ValueError("HF precision currently requires unpadded uniform-length batches")
        return original_forward(*args, **kwargs)

    model.forward = checked_forward
    model.ceno_precision_reference = dict(dtype="bfloat16", norm=norm_mode,
        cp_size=reference_cp_size, context_length=reference_context_length)
    return model


def load_model(model_dir, *, norm_mode="te-tuned", reference_cp_size=None,
               reference_context_length=None, device="cuda"):
    """Load the same implementation used by a configured AutoModel checkpoint."""
    configure_backend()
    import torch
    CENOConfig = _local_module("configuration_ceno").CENOConfig
    CENOForCausalLM = _local_module("modeling_ceno").CENOForCausalLM

    config = CENOConfig.from_pretrained(model_dir)
    model = CENOForCausalLM.from_pretrained(
        model_dir, config=config, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device).eval()
    return enable_training_precision(model, norm_mode=norm_mode,
        reference_cp_size=reference_cp_size, reference_context_length=reference_context_length)
