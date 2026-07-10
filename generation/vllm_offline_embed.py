#!/usr/bin/env python3
# Copyright (c) 2025-2026, CENO Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
vLLM offline embedding for the CENO DNA model.

Usage (in a container with vLLM installed, after running patch_vllm_for_dna.py):
    python3 generation/vllm_offline_embed.py <checkpoint_dir> [--prompts ATCG,GATTACA]
"""
import argparse
import os
import time

os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")

from vllm import LLM, PoolingParams


def parse_bool_flag(value):
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected one of: true/false/1/0/yes/no")


def build_llm(args):
    llm_kwargs = dict(
        model=args.model_dir,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        enable_prefix_caching=False,
        runner="pooling",
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    if args.kv_cache_dtype != "auto":
        llm_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
        llm_kwargs["calculate_kv_scales"] = True

    if args.convert != "none":
        try:
            return LLM(convert=args.convert, **llm_kwargs), args.convert
        except TypeError as exc:
            if "convert" not in str(exc):
                raise
            print("WARNING: current vLLM does not accept `convert=` in LLM(...); retrying without it.")

    return LLM(**llm_kwargs), None


def run_embed(llm, prompts, pooling_params):
    if hasattr(llm, "embed"):
        return llm.embed(prompts, pooling_params=pooling_params), "embed"
    return llm.encode(prompts, pooling_task="embed", pooling_params=pooling_params), "encode"


def preview_vector(values, n):
    clipped = values[:n]
    return ", ".join(f"{x:.6f}" for x in clipped)


def main():
    parser = argparse.ArgumentParser(description="DNA CENO offline embedding")
    parser.add_argument("model_dir", help="Path to HF checkpoint directory")
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Comma-separated DNA prompts (default: built-in examples)",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization method, e.g. 'fp8' (Hopper/Ada Lovelace only)",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="auto",
        help="KV cache dtype: 'auto' or 'fp8' (default: auto)",
    )
    parser.add_argument(
        "--convert",
        type=str,
        default="embed",
        help="vLLM model conversion mode. Use 'embed' by default, or 'none' to disable.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Optional output embedding dimension for Matryoshka-capable models",
    )
    parser.add_argument(
        "--use-activation",
        type=parse_bool_flag,
        default=None,
        help="Optional pooling use_activation flag: true/false",
    )
    parser.add_argument(
        "--show-n",
        type=int,
        default=8,
        help="How many embedding values to preview per sequence",
    )
    args = parser.parse_args()

    if args.prompts:
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    else:
        prompts = [
            "ATCGATCGATCGATCG",
            "GATTACAGATTACA",
            "ACGTACGTACGTACGT",
        ]

    pooling_kwargs = {}
    if args.dimensions is not None:
        pooling_kwargs["dimensions"] = args.dimensions
    if args.use_activation is not None:
        pooling_kwargs["use_activation"] = args.use_activation
    pooling_params = PoolingParams(**pooling_kwargs)

    llm, used_convert = build_llm(args)

    print(
        f"\n>>> Embedding {len(prompts)} sequences "
        f"(runner=pooling, convert={used_convert or 'implicit/none'}) ..."
    )
    t0 = time.time()
    outputs, api_name = run_embed(llm, prompts, pooling_params)
    elapsed = time.time() - t0

    total_prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    first_embedding = outputs[0].outputs.embedding
    embedding_dim = len(first_embedding)

    print("\n" + "=" * 60)
    print(
        f"Results  [api={api_name}, dim={embedding_dim}, "
        f"dimensions_arg={args.dimensions}, use_activation={args.use_activation}]"
    )
    print("=" * 60)
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        embedding = output.outputs.embedding
        prompt_tokens = len(output.prompt_token_ids)
        print(f"\n[{i + 1}/{len(outputs)}] prompt={prompt_tokens} tokens, embedding_dim={len(embedding)}")
        print(f"  Prompt:      {prompt[:80]!r}{'...' if len(prompt) > 80 else ''}")
        print(f"  Embedding[:{args.show_n}]: [{preview_vector(embedding, args.show_n)}]")
        print("-" * 60)

    print(f"\n{'=' * 60}")
    print(f"  Total time:          {elapsed:.2f} s")
    print(f"  Sequences:           {len(outputs)}")
    print(f"  Total prompt tokens: {total_prompt_tokens}")
    print(f"  Avg per sequence:    {elapsed / len(outputs):.2f} s")
    print(f"  Sequences / second:  {len(outputs) / elapsed:.2f}")
    print(f"  Embedding dim:       {embedding_dim}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
