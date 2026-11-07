"""Opt-in HF inference against the same BF16 Megatron reference as vLLM.

Use from the repository with the existing training image (Mamba 2.2.4 / FA2).
Defaults in exported checkpoints are unchanged. This adapter is inference-only,
single-GPU, BF16, unpadded uniform-length batches. Its prefill token budget is
bounded to avoid the old Mamba kernels' large-stride indexing paths.
"""
import contextlib
import contextvars
import functools
import os
import types

_CHUNK = 128


def _local_module(name):
    import importlib
    return importlib.import_module(f"{__package__}.{name}" if __package__ else name)


def configure_backend():
    """Set the cuBLASLt workspace limit; call before the first CUDA GEMM.

    The reference GEMMs run through cuBLASLt (see ``backend_scope``) with a
    1 KiB workspace, which is what fixes their algorithm and hence their bits.
    That is the only process-level setting. ``CUBLAS_WORKSPACE_CONFIG`` is not
    set: it only sizes the legacy cuBLAS handle workspace, which these GEMMs do
    not use (outputs are bit-identical with or without it), and under torch 2.7
    it adds ~30-40 us of host time to every GEMM call, the main cost of B=1
    decode. The torch backend flags are applied only inside precision-enabled
    forwards, so the surrounding program (e.g. an FP32 head trained with TF32)
    keeps its own settings.
    """
    os.environ["CUBLASLT_WORKSPACE_SIZE"] = "1"


@contextlib.contextmanager
def backend_scope():
    """Apply the reference GEMM flags inside the block and restore them on exit."""
    import torch
    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    has_cuda = torch.version.cuda is not None
    saved = (matmul.allow_bf16_reduced_precision_reduction,
             matmul.allow_fp16_reduced_precision_reduction,
             matmul.allow_tf32, cudnn.allow_tf32,
             torch.backends.cuda.preferred_blas_library() if has_cuda else None)
    matmul.allow_bf16_reduced_precision_reduction = False
    matmul.allow_fp16_reduced_precision_reduction = False
    matmul.allow_tf32 = False
    cudnn.allow_tf32 = False
    if has_cuda:
        torch.backends.cuda.preferred_blas_library("cublaslt")
    try:
        yield
    finally:
        (matmul.allow_bf16_reduced_precision_reduction,
         matmul.allow_fp16_reduced_precision_reduction,
         matmul.allow_tf32, cudnn.allow_tf32) = saved[:4]
        if has_cuda:
            torch.backends.cuda.preferred_blas_library(saved[4])


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
                state["states_in"] = None  # derived from "initial"; rebuild lazily
            return result

        cache.reorder_cache = reorder
    return cache._ceno_chunks


def _install_scan_hooks(module):
    """Namespace hooks are inactive unless a precision-enabled mixer sets context."""
    import importlib
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
        state = {"initial": initial, "pos": count, "states_in": None}
        values = {"x": x, "dt": dt, "B": B, "C": C, "z": z}
        for key, value in values.items():
            buffer = torch.zeros((batch, _CHUNK, *value.shape[2:]),
                                 device=x.device, dtype=value.dtype)
            if count:
                buffer[:, :count].copy_(value[:, -count:])
            state[key] = buffer
        _cache_buffers(cache)[mixer.layer_idx] = state
        return result

    # The combined scan's own building blocks. Off the chunk boundary only the
    # output row is needed, so chunk_state/state_passing (which only produce the
    # final state) are skipped; the remaining calls receive exactly the inputs the
    # combined forward would pass them, so the row is bit-identical.
    parts = importlib.import_module(scan.__module__)
    fast = all(hasattr(parts, name) for name in
               ("_chunk_cumsum_fwd", "_bmm_chunk_fwd", "_chunk_scan_fwd", "_state_passing_fwd"))

    def chunk_row(mixer, state, pos):
        if state.get("states_in") is None:
            # Entering-chunk state exactly as _state_passing_fwd emits it for chunk 0
            # (FP32 initial stored in C's dtype).
            initial = state["initial"]
            batch, heads, dim, dstate = initial.shape
            flat = initial.reshape(batch, heads, dim * dstate)
            out, _ = parts._state_passing_fwd(
                torch.zeros((batch, 1, heads, dim * dstate), device=flat.device, dtype=torch.float32),
                torch.zeros((batch, heads, 1), device=flat.device, dtype=torch.float32),
                initial_states=flat, chunk_size=_CHUNK, out_dtype=state["C"].dtype)
            state["states_in"] = out.view(batch, 1, heads, dim, dstate)
        dA_cumsum, dt = parts._chunk_cumsum_fwd(
            state["dt"], state["A"], _CHUNK, dt_bias=mixer.dt_bias, dt_softplus=True,
            dt_limit=mixer.time_step_limit)
        CB = parts._bmm_chunk_fwd(state["C"], state["B"], _CHUNK, seq_idx=None,
                                  output_dtype=torch.float32)
        D = mixer.D if mixer.D.stride(-1) == 1 else mixer.D.contiguous()
        out, _ = parts._chunk_scan_fwd(CB, state["x"], dt, dA_cumsum, state["C"],
                                       state["states_in"], D=D, z=state["z"], seq_idx=None)
        return out[:, pos]

    def update_from_chunk(state_tensor, x, dt, A, B, C, D, *args, **kwargs):
        active = context.get()
        if active is None:
            return update(state_tensor, x, dt, A, B, C, D, *args, **kwargs)
        mixer, cache = active
        state = _cache_buffers(cache)[mixer.layer_idx]
        state["decoded"] = True  # decode kernels have run eagerly once (graph warm-up)
        pos = state["pos"]
        # The recurrent update's output was never used and state_tensor is only
        # read back at the chunk boundary below, so it is not run per token.
        for key, value in (("x", x), ("dt", dt[:, :, 0]), ("B", B), ("C", C),
                           ("z", kwargs["z"])):
            if pos == 0:
                state[key].zero_()
            state[key][:, pos].copy_(value)
        if "A" not in state:
            state["A"] = -torch.exp(mixer.A_log.float())
        if pos == _CHUNK - 1 or not fast:
            y, final = scan(state["x"], state["dt"], state["A"],
                            state["B"], state["C"], chunk_size=_CHUNK, D=mixer.D,
                            z=state["z"], dt_bias=mixer.dt_bias, dt_softplus=True,
                            dt_limit=mixer.time_step_limit, initial_states=state["initial"],
                            return_final_states=True)
            y = y[:, pos]
        else:
            y = chunk_row(mixer, state, pos)
        if pos == _CHUNK - 1:
            state["initial"].copy_(final)
            state_tensor.copy_(final)
            state["states_in"] = None
        state["pos"] = (pos + 1) % _CHUNK
        return y

    module.mamba_chunk_scan_combined = scan_with_cache
    module.selective_state_update = update_from_chunk
    module._ceno_scan_context = context
    module._ceno_scan_parts = parts if fast else None
    return context


class _DecodeGraph:
    """Batch-1 cached decode replayed as CUDA graphs, one per span between attention layers.

    Every kernel is the one the eager decode step launches, with the same shapes and
    inputs, so replay is bit-identical; only the host work is removed. Three eager
    pieces remain: the attention mixers (growing KV cache), the chunk-boundary step
    (full scan, handled by the caller falling back to eager) and the chunk-buffer
    reset at chunk position 0. Two data-dependent host decisions are moved onto the
    device: the chunk position (an index tensor) and the MoE expert loop (every
    expert runs on the single token; a non-selected expert's result is discarded by
    torch.where, a selected one goes through the same index_add_ as eagerly).
    """

    _KEYS = ("x", "dt", "B", "C", "z")

    def __init__(self, model, modeling, cache):
        import torch
        self.model, self.modeling, self.cache = model, modeling, cache
        self.parts = modeling._ceno_scan_parts
        self.layers = model.backbone.layers
        self.chunks = cache._ceno_chunks
        self.mamba = [i for i, b in enumerate(self.layers) if b.block_type == "mamba"]
        self.attention = [i for i, b in enumerate(self.layers) if b.block_type == "attention"]
        self.inference = torch.is_inference_mode_enabled()
        self.key = self._key()
        device = model.device
        self.ids = torch.zeros((1, 1), device=device, dtype=torch.long)
        self.pos = torch.zeros((1,), device=device, dtype=torch.long)
        self.row0 = torch.zeros((1,), device=device, dtype=torch.long)
        hidden = model.config.hidden_size
        self.attn_out = [torch.zeros((1, 1, hidden), device=device, dtype=model.dtype)
                         for _ in self.attention]
        self.attn_in, self.attn_res, self.graphs = [None] * len(self.attention), [None] * len(self.attention), []
        self.logits = None
        self._capture()

    def _key(self):
        import torch
        ptrs = [self.cache.conv_states[i].data_ptr() for i in self.mamba]
        for i in self.mamba:
            state = self.chunks[i]
            ptrs += [state[k].data_ptr() for k in self._KEYS + ("initial", "A")]
        return (torch.is_inference_mode_enabled(), tuple(ptrs))

    def valid(self, cache):
        return cache is self.cache and self._key() == self.key

    # ----- graph bodies: same module calls / kernels as the eager decode path -----
    def _mamba(self, i, hidden_states):
        import torch
        mixer, cache, state = self.layers[i].mixer, self.cache, self.chunks[i]
        hidden_states = mixer._maybe_apply_in_proj_norm(hidden_states)
        projected_states = mixer.in_proj(hidden_states)
        batch_size = hidden_states.shape[0]
        groups_time_state_size = mixer.n_groups * mixer.ssm_state_size
        d_mlp = (projected_states.shape[-1] - 2 * mixer.intermediate_size
                 - 2 * groups_time_state_size - mixer.num_heads) // 2
        _, _, gate, hidden_states_B_C, dt = projected_states.squeeze(1).split(
            [d_mlp, d_mlp, mixer.intermediate_size, mixer.conv_dim, mixer.num_heads], dim=-1)
        hidden_states_B_C = self.modeling.causal_conv1d_update(
            hidden_states_B_C, cache.conv_states[i], mixer.conv1d.weight.squeeze(1),
            mixer.conv1d.bias, mixer.activation)
        x, B, C = torch.split(hidden_states_B_C,
                              [mixer.intermediate_size, groups_time_state_size, groups_time_state_size], dim=-1)
        values = (x.view(batch_size, mixer.num_heads, mixer.head_dim), dt,
                  B.view(batch_size, mixer.n_groups, -1), C.view(batch_size, mixer.n_groups, -1),
                  gate.view(batch_size, mixer.num_heads, mixer.head_dim))
        for key, value in zip(self._KEYS, values):
            state[key].index_copy_(1, self.pos, value.unsqueeze(1))
        parts = self.parts
        initial = state["initial"]
        batch, heads, dim, dstate = initial.shape
        flat = initial.reshape(batch, heads, dim * dstate)
        states_in, _ = parts._state_passing_fwd(
            torch.zeros((batch, 1, heads, dim * dstate), device=flat.device, dtype=torch.float32),
            torch.zeros((batch, heads, 1), device=flat.device, dtype=torch.float32),
            initial_states=flat, chunk_size=_CHUNK, out_dtype=state["C"].dtype)
        states_in = states_in.view(batch, 1, heads, dim, dstate)
        dA_cumsum, dt = parts._chunk_cumsum_fwd(
            state["dt"], state["A"], _CHUNK, dt_bias=mixer.dt_bias, dt_softplus=True,
            dt_limit=mixer.time_step_limit)
        CB = parts._bmm_chunk_fwd(state["C"], state["B"], _CHUNK, seq_idx=None, output_dtype=torch.float32)
        D = mixer.D if mixer.D.stride(-1) == 1 else mixer.D.contiguous()
        out, _ = parts._chunk_scan_fwd(CB, state["x"], dt, dA_cumsum, state["C"], states_in,
                                       D=D, z=state["z"], seq_idx=None)
        y = out.index_select(1, self.pos).view(batch_size, mixer.num_heads * mixer.head_dim)
        y = mixer.norm(y, None)
        return mixer.out_proj(y)[:, None, ...]

    def _moe(self, moe, hidden_states):
        import torch
        batch_size, seq_len, hidden_dim = hidden_states.shape
        router_softmax = getattr(moe, "_ceno_router_softmax", torch.nn.functional.softmax)
        flat = hidden_states.view(-1, hidden_dim)
        if str(getattr(moe.config, "moe_router_dtype", "")).lower() == "fp32":
            router_logits = torch.nn.functional.linear(flat.to(torch.float32), moe.gate.weight.to(torch.float32), bias=None)
        else:
            router_logits = moe.gate(flat).to(torch.float32)
        topk_logits, selected = torch.topk(router_logits, moe.top_k, dim=-1)
        weights = router_softmax(topk_logits, dim=-1)
        final = torch.zeros_like(flat)
        for expert_idx in range(moe.num_experts):
            hit = selected[0] == expert_idx                       # (top_k,)
            weight = torch.where(hit, weights[0], 0.).sum().view(1, 1).to(flat.dtype)
            out = moe.experts[expert_idx](flat, per_token_scale=weight)
            added = final.index_add(0, self.row0, out.to(hidden_states.dtype))
            final = torch.where(hit.any(), added, final)
        return final.view(batch_size, seq_len, hidden_dim)

    def _span(self, start, end, hidden_states):
        for i in range(start, end):
            block = self.layers[i]
            if block.block_type == "mamba":
                residual = hidden_states
                if block.norm is not None:
                    hidden_states = block.norm(hidden_states.to(dtype=block.norm.weight.dtype))
                if block.residual_in_fp32:
                    residual = residual.to(self._fp32)
                hidden_states = residual + self._mamba(i, hidden_states)
            elif block.block_type in ("moe", "mlp"):
                input_ln_out = block.norm(hidden_states.to(dtype=block.norm.weight.dtype))
                residual = input_ln_out
                if block.residual_in_fp32:
                    residual = residual.to(self._fp32)
                pre = block.pre_mlp_norm(input_ln_out.to(dtype=block.pre_mlp_norm.weight.dtype))
                out = self._moe(block.mixer, pre) if block.block_type == "moe" else block.mixer(pre)
                hidden_states = residual + out
            else:
                raise RuntimeError("attention layers are span boundaries")
        return hidden_states

    def _segment(self, k):
        """Span k: after attention k-1 (or the embedding) up to attention k's input norm."""
        bounds = [-1] + self.attention + [len(self.layers)]
        start, end = bounds[k] + 1, bounds[k + 1]
        if k == 0:
            hidden_states = self.model.backbone.embeddings(self.ids)
        else:
            block = self.layers[bounds[k]]
            hidden_states = self.attn_res[k - 1] + self.attn_out[k - 1]
            hidden_states = block.pre_mlp_norm(hidden_states.to(dtype=block.pre_mlp_norm.weight.dtype))
        hidden_states = self._span(start, end, hidden_states)
        if k < len(self.attention):
            block = self.layers[end]
            residual = hidden_states
            self.attn_in[k] = block.norm(hidden_states.to(dtype=block.norm.weight.dtype))
            self.attn_res[k] = residual.to(self._fp32) if block.residual_in_fp32 else residual
        else:
            hidden_states = self.model.backbone.norm_f(hidden_states)
            lm_head = self.model.lm_head
            self.logits = lm_head(hidden_states.to(lm_head.weight.dtype)).float()

    def _capture(self):
        import torch
        self._fp32 = torch.float32
        segments = len(self.attention) + 1
        # The warm-up really runs the bodies: it shifts the conv states and writes the
        # chunk buffers. Snapshot everything it can write and restore it afterwards.
        mutable = [self.cache.conv_states[i] for i in self.mamba]
        mutable += [self.chunks[i][key] for i in self.mamba for key in self._KEYS]
        snapshot = [t.clone() for t in mutable]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for k in range(segments):
                self._segment(k)
        torch.cuda.current_stream().wait_stream(stream)
        for tensor, saved in zip(mutable, snapshot):
            tensor.copy_(saved)
        del snapshot
        pool = torch.cuda.graph_pool_handle()
        for k in range(segments):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                self._segment(k)
            self.graphs.append(graph)
        torch.cuda.synchronize()

    def step(self, input_ids, pos):
        if pos == 0:
            for i in self.mamba:
                for key in self._KEYS:
                    self.chunks[i][key].zero_()
        self.ids.copy_(input_ids)
        self.pos.fill_(pos)
        for k, graph in enumerate(self.graphs):
            graph.replay()
            if k < len(self.attention):
                mixer = self.layers[self.attention[k]].mixer
                out = mixer(self.attn_in[k], attention_mask=None, past_key_value=self.cache,
                            use_cache=True, cache_position=None)[0]
                self.attn_out[k].copy_(out)
        for i in self.mamba:
            self.chunks[i]["pos"] = pos + 1
        return self.logits.clone()


def _decode_graph_step(model, modeling, kwargs):
    """Return logits from the graph path, or None to run the eager forward."""
    if os.environ.get("CENO_HF_DECODE_GRAPH", "1") == "0" or getattr(modeling, "_ceno_scan_parts", None) is None:
        return None
    if any(kwargs.get(k) is not None for k in ("inputs_embeds", "labels", "seq_idx")) or any(
            kwargs.get(k) for k in ("output_attentions", "output_hidden_states")):
        return None
    if kwargs.get("use_cache") is False or (kwargs.get("use_cache") is None and not model.config.use_cache):
        return None
    ids = kwargs.get("input_ids")
    cache = kwargs.get("cache_params") or kwargs.get("past_key_values")
    position = kwargs.get("cache_position")
    if (ids is None or tuple(ids.shape) != (1, 1) or cache is None or position is None
            or position.numel() != 1 or model.config.output_hidden_states or model.config.output_attentions):
        return None
    chunks = getattr(cache, "_ceno_chunks", None)
    layers = model.backbone.layers
    mamba = [i for i, b in enumerate(layers) if b.block_type == "mamba"]
    if not mamba or not chunks or any(i not in chunks or not chunks[i].get("decoded") for i in mamba):
        return None
    positions = {chunks[i]["pos"] for i in mamba}
    if len(positions) != 1 or any(len(cache.key_cache[i].shape) != 4 for i in cache.attention_layers):
        return None
    pos = positions.pop()
    if pos == _CHUNK - 1:
        return None  # chunk boundary: full scan in the eager path
    runner = getattr(cache, "_ceno_decode_graph", None)
    with backend_scope():
        if runner is None or not runner.valid(cache):
            runner = cache._ceno_decode_graph = None
            if pos == 0:  # the eager reset happens in step(); capture after it
                for i in mamba:
                    for key in _DecodeGraph._KEYS:
                        chunks[i][key].zero_()
            runner = cache._ceno_decode_graph = _DecodeGraph(model, modeling, cache)
        return runner.step(ids, pos)


def _attention(q, k, v, causal):
    """Ordinary FA2 always; dense FA2 can change arithmetic with query length."""
    import torch
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    # Build [0, n] on the device: torch.tensor(..., device=cuda) copies from
    # pageable host memory and synchronizes the stream on every call.
    steps = torch.arange(2, device=q.device, dtype=torch.int32)
    cuq, cuk = steps * len(q), steps * len(k)
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
    norm_module = _local_module("ceno_norm")
    te_rms_norm, te_rms_norm_eager = norm_module.te_rms_norm, norm_module._te_rms_norm_impl

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
    mark = getattr(modeling, "_mark_backend_scope", None)
    if mark is None:
        raise ValueError("HF precision requires a modeling file with scoped backend flags")
    mark(model)
    config.ceno_mamba_gate_in_ssm = True
    config.ceno_rmsnorm_mode = norm_mode
    config.ceno_reference_cp_size = reference_cp_size
    config.ceno_reference_context_length = reference_context_length
    _pin_scan()
    context = _install_scan_hooks(modeling)
    def norm(self, x):
        # Same Triton kernel; the registered op is only needed under torch.compile.
        op = te_rms_norm if torch.compiler.is_compiling() else te_rms_norm_eager
        return op(x, self.weight, self.variance_epsilon, norm_mode)

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
        if len(args) <= 1 and not (args and "input_ids" in kwargs):
            graph_kwargs = dict(kwargs, input_ids=args[0]) if args else kwargs
            logits = _decode_graph_step(model, modeling, graph_kwargs)
            if logits is not None:
                cache = kwargs.get("cache_params") or kwargs.get("past_key_values")
                if kwargs.get("return_dict", model.config.use_return_dict) is False:
                    return (logits, cache)
                return modeling.CENOCausalLMOutput(logits=logits, cache_params=cache,
                                                        past_key_values=cache)
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
