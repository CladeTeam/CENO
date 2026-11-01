#!/usr/bin/env python3
"""
vLLM offline inference for DNA NemotronH model.

Usage (in container, after running patch_vllm_for_dna.py):
    python3 generation/vllm_offline_infer.py <checkpoint_dir> [--prompts ATCG,GATTACA]
"""
import argparse
import json
from pathlib import Path
import os
import sys
import time

os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")
# Read by vLLM at import time. Explicit VLLM_BATCH_INVARIANT=0 keeps the legacy path.
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
_checkpoint_precision = False
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    config_path = Path(sys.argv[1]) / "config.json"
    if config_path.is_file():
        _checkpoint_precision = bool(json.loads(config_path.read_text()).get("ceno_training_precision"))
if _checkpoint_precision or "--training-precision" in sys.argv:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# Configure Triton before importing the engine/compilation machinery.
from vllm.model_executor.layers.ceno_mamba import configure_invariant_scan

configure_invariant_scan()

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="DNA NemotronH offline inference")
    parser.add_argument("model_dir", help="Path to HF checkpoint directory")
    parser.add_argument("--prompts", type=str, default=None,
                        help="Comma-separated DNA prompts (default: built-in examples)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=8, help="Maximum concurrent requests")
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    parser.add_argument("--quantization", type=str, default=None,
                        help="Quantization method, e.g. 'fp8' (Hopper/Ada Lovelace only)")
    parser.add_argument("--kv-cache-dtype", type=str, default="auto",
                        help="KV cache dtype: 'auto' or 'fp8' (default: auto)")
    parser.add_argument("--mamba-ssm-cache-dtype", choices=("float32", "auto", "float16"),
                        default="float32",
                        help="SSM recurrent state precision; FP32 prevents long-decode BF16 drift")
    parser.add_argument("--training-precision", action="store_true", default=_checkpoint_precision,
                        help="CENO training arithmetic (enabled by checkpoint configuration); requires prepared TE norm and "
                             "training-gate runtime. Uses extra KV staging memory and bandwidth.")
    args = parser.parse_args()
    if args.max_num_seqs < 1:
        parser.error("--max-num-seqs must be positive")

    # Prompts
    if args.prompts:
        prompts = [p.strip() for p in args.prompts.split(",")]
    else:
        prompts = [
            "ATCGATCGATCGATCG",
            "GATTACAGATTACA",
            "ACGTACGTACGTACGT",
        ]

    # Sampling params
    sp_kwargs = dict(temperature=args.temperature, max_tokens=args.max_tokens)
    if args.top_p is not None:
        sp_kwargs["top_p"] = args.top_p
    if args.top_k is not None:
        sp_kwargs["top_k"] = args.top_k
    sampling_params = SamplingParams(**sp_kwargs)

    # Create LLM
    llm_kwargs = dict(
        model=args.model_dir,
        trust_remote_code=True,
        dtype="bfloat16",
        attention_config={"backend": os.environ["VLLM_ATTENTION_BACKEND"]},
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        enable_prefix_caching=False,
        mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
        # Keep graph replay speed without compiler-dependent BF16 arithmetic.
        compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    if args.kv_cache_dtype != "auto":
        llm_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
        llm_kwargs["calculate_kv_scales"] = True
    if args.training_precision:
        from vllm.model_executor.layers.ceno_precision import install_training_precision
        install_training_precision()
        llm_kwargs.update(attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 2},
                          async_scheduling=False, max_num_seqs=min(args.max_num_seqs, len(prompts)))
    llm = LLM(**llm_kwargs)

    # Generate with timing
    print(f"\n>>> Generating {len(prompts)} sequences, max_tokens={args.max_tokens} ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0

    # Stats
    total_prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    total_gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    # Print
    print("\n" + "=" * 60)
    print(f"Results  [temp={args.temperature}, top_p={args.top_p}, "
          f"top_k={args.top_k}, max_tokens={args.max_tokens}]")
    print("=" * 60)
    for i, output in enumerate(outputs):
        gen_tokens = len(output.outputs[0].token_ids)
        prompt_tokens = len(output.prompt_token_ids)
        print(f"\n[{i+1}/{len(outputs)}] prompt={prompt_tokens} tokens, "
              f"generated={gen_tokens} tokens")
        print(f"  Prompt:  {output.prompt[:80]!r}{'...' if len(output.prompt) > 80 else ''}")
        print(f"  Output:  {output.outputs[0].text[:80]!r}{'...' if len(output.outputs[0].text) > 80 else ''}")
        print("-" * 60)

    print(f"\n{'=' * 60}")
    print(f"  Total time:          {elapsed:.2f} s")
    print(f"  Sequences:           {len(outputs)}")
    print(f"  Total prompt tokens: {total_prompt_tokens}")
    print(f"  Total gen tokens:    {total_gen_tokens}")
    print(f"  Throughput:          {total_gen_tokens / elapsed:.1f} tokens/s")
    print(f"  Avg per sequence:    {elapsed / len(outputs):.2f} s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
