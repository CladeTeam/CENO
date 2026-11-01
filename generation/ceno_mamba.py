"""Fix native Mamba reduction configurations for reproducible CENO inference.

Autotuning can select numerically different reduction tiles in fresh processes.
Only enabled with vLLM's explicit batch-invariant mode; kernels and cache
handling remain native vLLM implementations.
"""

_configured = False


def configure_invariant_scan():
    global _configured
    if _configured:
        return
    from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
    if not vllm_is_batch_invariant():
        return
    from vllm.model_executor.layers.mamba.ops import (
        ssd_bmm, ssd_chunk_scan, ssd_chunk_state, ssd_state_passing,
    )

    matrix = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}
    choices = (
        (ssd_bmm, "_bmm_chunk_fwd_kernel", matrix),
        (ssd_chunk_scan, "_chunk_scan_fwd_kernel", matrix),
        (ssd_chunk_state, "_chunk_cumsum_fwd_kernel", {"BLOCK_SIZE_H": 4}),
        (ssd_chunk_state, "_chunk_state_fwd_kernel", matrix),
        (ssd_state_passing, "_state_passing_fwd_kernel", {"BLOCK_SIZE": 256}),
    )
    selected = []
    for module, name, wanted in choices:
        kernel = getattr(module, name)
        matches = [config for config in kernel.configs if config.kwargs == wanted]
        if len(matches) != 1:
            raise RuntimeError(f"Unsupported native Mamba kernel configuration: {name}")
        selected.append((kernel, matches))
    for kernel, configs in selected:
        kernel.configs[:] = configs
        kernel.cache.clear()
    _configured = True
