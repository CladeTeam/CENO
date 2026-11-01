# CENO inference

Use one current implementation: this directory, its nine `ceno_*.py` helpers,
and `third_party/TransformerEngine-LICENSE`. Use the generation scripts in this repository.

## Environment and generation

Use vLLM **0.15.1**. `requirements-vllm.txt` and the Dockerfile agree on this
version; upgrading to another release requires a matching integration.
The full exact-reference evidence was collected on H200 with Torch 2.9.1+cu129.
A matching package version alone is not a bitwise guarantee on every GPU/build.

Create an independent runtime copy of a downloaded base CENO checkpoint:

```bash
set -euo pipefail
RUNTIME=$(mktemp -d /tmp/ceno-runtime.XXXXXX)
cp -aL /path/to/checkpoint/. "$RUNTIME/"
python3 generation/patch_vllm_for_dna.py "$RUNTIME"
python3 generation/vllm_offline_infer.py "$RUNTIME" \
  --prompts ATCGATCGATCG,GATTACAGATTACA --max-tokens 256 --temperature 0.8
```

Current checkpoint configuration selects the inference arithmetic automatically.
The patcher modifies the runtime and container-local vLLM, never source weights.
Do not use hardlinked runtime copies. Keep all generation helpers beside the
patcher. CENO-P is an HF MSA scorer, not a vLLM generation checkpoint.

## Numerical contract

The corrected path uses FP32 Mamba state, training-order gating and MoE weighting,
fixed scan reductions, cache-owned chunk decode, fixed TE RMSNorm, and ordinary
FA2 attention with an explicit CP reference. CP1 and CP8 Megatron references are
not bitwise interchangeable. `ceno_reference_context_length` is fixed per model,
never derived from the current batch or prefix.

Current precision support: BF16 CENO widths 512/1024, Mamba heads 16/32,
attention heads 8/16, TP=PP=1, no quantization, prefix caching or speculation.
Decode CUDA Graphs remain enabled; asynchronous scheduling and whole-model
compilation are disabled. The CLI defaults to eight simultaneous requests; use `--max-num-seqs` to change it.
Other architectures keep their ordinary compatible implementation and are not
labeled training-precision validated. CENO-P uses the separate HF MSA path.

The precision path costs additional KV staging memory and scan work. Temperature
changes sampling; it does not repair raw-logit disagreement or guarantee
biological quality. Seeded sampling is not expected to produce identical random
sequences across different inference engines.

## HF and validation

HF standalone exports include their own relative-import helpers; vLLM is not an
HF dependency. See [HF loading](../ceno_model/README.md). The HF precision path is single-GPU, BF16,
inference-only, unpadded equal-length batches, with a 131072-token prefill budget.
Use vLLM for larger prefills. The embedding CLI is retained but generation's
bitwise-parity claims do not apply to embedding pooling.

Existing c2 evidence: CP1 continuous 131072-step decode and CP8 1044480-token prefix +4096
steps (1M tokens total) match their fixed Megatron BF16 references; HF evidence covers 4874 steps
and a 131k prefix +64 steps. These are specific checks, not a claim that every
published checkpoint has the same coverage. The README records the tested limits; historical diagnostic bundles are not required.

## Measured performance

Same H200, same 1B c2 weights, one full-length warm-up and median of three runs.
Throughput includes prefill and decode, excludes model loading/compilation;
batch-eight values count all generated tokens. The baseline is the previous
shared implementation, which had numerical errors.

| Prefix tokens | Batch | Generated tokens per request | Previous tokens/s | Current tokens/s | Change |
|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 1024 | 441.75 | 361.27 | -18.2% |
| 512 | 8 | 1024 | 2846.00 | 2451.51 | -13.9% |
| 8192 | 1 | 1024 | 373.96 | 293.57 | -21.5% |
| 8192 | 8 | 1024 | 2335.82 | 1726.41 | -26.1% |
| 131072 | 1 | 256 | 84.53 | 74.44 | -11.9% |

At the 131072-token prefix, decode alone improves from 110.87 to 118.14 tokens/s
(+6.6%); time to first token increases from 0.728 to 1.281 seconds. These numbers
are workload-specific and do not imply unchanged throughput for all models.
