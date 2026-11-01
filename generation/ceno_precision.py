"""Opt in to the validated CENO training arithmetic on vLLM 0.15.1.

Call before constructing LLM. Mamba's extra state lives in managed cache pages;
attention repacks those pages into ordinary FA2 inputs. The extra workspace and
copying are a deliberate cost of this precision path. Defaults are unchanged.
"""
import functools
import ast
import importlib.metadata
import inspect
import textwrap

_installed = False


def validate_config(config):
    """Reject configurations outside this path's tested numerical contract."""
    model, cache, parallel = config.model_config, config.cache_config, config.parallel_config
    hf = model.hf_config
    expected = {
        "hidden_size": (512, 1024),
        "mamba_num_heads": (16, 32),
        "mamba_head_dim": 64,
        "ssm_state_size": 128,
        "n_groups": 8,
        "chunk_size": 128,
    }
    for name, value in expected.items():
        if getattr(hf, name, None) not in (value if isinstance(value, tuple) else (value,)):
            raise ValueError(f"Training precision requires CENO {name}={value}")
    if (getattr(hf, "moe_router_pre_softmax", False)
            or getattr(hf, "mlp_use_swiglu", False)
            or getattr(hf, "mlp_hidden_act", "relu2") not in ("relu2", "squared_relu")
            or getattr(hf, "num_experts", 8) != 8
            or getattr(hf, "moe_top_k", 2) != 2
            or getattr(hf, "n_shared_experts", 0) not in (None, 0)):
        raise ValueError("Training precision requires post-topk ReLU2 experts, E=8, top-k=2")
    if tuple(getattr(hf, "time_step_limit", (0., float("inf")))) not in ((0., float("inf")), (0., 1e30)):
        raise ValueError("Training precision requires the unbounded default Mamba dt interval")
    if not getattr(hf, "ceno_mamba_gate_in_ssm", False):
        raise ValueError("Prepare the runtime copy with --mamba-gate-mode training")
    if getattr(hf, "ceno_rmsnorm_mode", None) not in ("te-general", "te-tuned"):
        raise ValueError("Prepare the runtime copy with an explicit TE --rmsnorm-mode")
    cp = getattr(hf, "ceno_reference_cp_size", 1)
    if cp not in (1, 8):
        raise ValueError("The validated reference CP sizes are 1 and 8")
    if cp == 8:
        if (getattr(hf, "num_attention_heads", None) not in (8, 16)
                or getattr(hf, "num_key_value_heads", None) != hf.num_attention_heads
                or (getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads) not in (64, 128)):
            raise ValueError("CP8 precision currently requires matching 8/16 query/KV heads of width 64/128")
        length = getattr(hf, "ceno_reference_context_length", None)
        if not isinstance(length, int) or length <= 0 or length % 1024:
            raise ValueError("CP8 requires an explicit reference context divisible by 1024")
        if model.max_model_len > length:
            raise ValueError("Serving context exceeds the fixed CP reference context")
        if hf.ceno_rmsnorm_mode != "te-tuned":
            raise ValueError("CP8 reference requires te-tuned RMSNorm")
    if str(model.dtype) != "torch.bfloat16" or getattr(model, "quantization", None):
        raise ValueError("Training precision requires unquantized BF16 weights")
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise ValueError("Training precision currently requires TP=PP=1")
    if config.speculative_config or cache.enable_prefix_caching or cache.mamba_cache_mode == "all":
        raise ValueError("Training precision does not support speculation or prefix caching")
    if cache.mamba_ssm_cache_dtype != "float32":
        raise ValueError("Training precision requires mamba_ssm_cache_dtype='float32'")
    backend = config.attention_config.backend
    if (getattr(backend, "name", str(backend)) != "FLASH_ATTN"
            or config.attention_config.flash_attn_version != 2):
        raise ValueError("Training precision requires FLASH_ATTN with flash_attn_version=2")
    if config.scheduler_config.async_scheduling:
        raise ValueError("Training precision currently requires async_scheduling=False")
    mode = config.compilation_config.mode
    if getattr(mode, "value", mode) != 0:
        raise ValueError("Training precision requires compilation_config mode=0")


def install_training_precision():
    """Install once before LLM startup; validate the actual engine configuration."""
    global _installed
    if _installed:
        return
    if importlib.metadata.version("vllm") != "0.15.1":
        raise RuntimeError("Training precision requires the validated vLLM 0.15.1 runtime")
    import vllm.envs as envs
    if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
        raise ValueError("Set VLLM_ENABLE_V1_MULTIPROCESSING=0 before importing vLLM; "
                         "this precision installer currently runs in one process")
    from vllm.model_executor.layers import ceno_attention, ceno_chunk
    from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
    from vllm.model_executor.layers.ceno_mamba import configure_invariant_scan
    from vllm.model_executor.layers.mamba import mamba_mixer2 as mm
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
    from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder

    if not vllm_is_batch_invariant():
        raise ValueError("Set VLLM_BATCH_INVARIANT=1 before importing vLLM")
    configure_invariant_scan()
    source = textwrap.dedent(inspect.getsource(mm.MambaMixer2.conv_ssm_forward))
    if "ceno_gate_in_ssm" not in source:
        raise RuntimeError("Run the matching CENO patcher before enabling training precision")

    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError(f"Unsupported vLLM Mamba source: {old}")
        source = source.replace(old, new)

    replace("conv_state = self_kv_cache[0].transpose(-1, -2)",
            "conv_state = self_kv_cache[0][:, :self.conv_kernel_size-1].transpose(-1, -2)")
    replace("ssm_state = self_kv_cache[1]",
            "ssm_state = self_kv_cache[1][..., :self.ssm_state_size]")
    replace("varlen_states = mamba_chunk_scan_combined_varlen(",
            "varlen_states = functools.partial(ceno_chunk.prefill, self_kv_cache, "
            "state_indices_tensor_p, has_initial_states_p, self.conv_kernel_size-1)(")
    replace('dt_limit=(0.0, float("inf")),', 'dt_limit=self.ceno_dt_limit,')
    replace("if has_decode:", """if has_decode:
        ceno_chunk.reset_new_decode(self_kv_cache, state_indices_tensor_d,
                                   attn_metadata.ceno_seq_lens_d)""")
    source += """
        ceno_chunk.decode(self_kv_cache, state_indices_tensor_d, hidden_states_d, dt_d,
                          self.A, B_d, C_d, self.D,
                          projected_states[:num_decodes, :self.tped_intermediate_size].view(
                              num_decodes, self.num_heads, self.head_dim),
                          self.dt_bias,
                          preallocated_ssm_out_d.view(num_decodes, -1, self.head_dim),
                          self.conv_kernel_size-1, dt_limit=self.ceno_dt_limit)
"""
    # The chunk decoder owns the relevant state and replaces the native output.
    # Avoid doing an unused recurrent update before the exact SSD row calculation.
    tree = ast.parse(source)
    unused = [node for node in ast.walk(tree) if isinstance(node, ast.Expr)
              and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Name)
              and node.value.func.id == "selective_state_update"]
    if len(unused) != 1:
        raise RuntimeError("Unsupported native decode update layout")
    node = unused[0]
    lines = source.splitlines(keepends=True)
    source = "".join(lines[:node.lineno-1] + lines[node.end_lineno:])
    scope = dict(vars(mm), functools=functools, ceno_chunk=ceno_chunk)
    exec(compile(source, "<CENO cache-owned chunk decode>", "exec"), scope)
    original_shape = mm.MambaMixer2.get_state_shape
    original_model_shape = NemotronHForCausalLM.get_mamba_state_shape_from_config
    original_build = Mamba2AttentionMetadataBuilder.build

    def state_shape(self):
        return ceno_chunk.shapes(*original_shape(self))

    def model_shape(cls, config):
        validate_config(config)
        return ceno_chunk.shapes(*original_model_shape(config))

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        result = original_build(self, common_prefix_len, common_attn_metadata, fast_build)
        result.ceno_seq_lens_d = common_attn_metadata.seq_lens[:result.num_decode_tokens]
        return result

    mm.MambaMixer2.conv_ssm_forward = scope["conv_ssm_forward"]
    mm.MambaMixer2.get_state_shape = state_shape
    NemotronHForCausalLM.get_mamba_state_shape_from_config = classmethod(model_shape)
    Mamba2AttentionMetadataBuilder.build = build
    ceno_attention.install()
    _installed = True
