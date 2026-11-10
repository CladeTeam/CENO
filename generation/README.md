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

For a custom Python integration, reuse `engine_kwargs()` from
`vllm_offline_infer.py` (put `--exact` in `sys.argv` before importing the module
to select the exact path). A direct `LLM(...)` call after patching does not
inherit the CLI setup.

## Engine modes

`vllm_offline_infer.py` has two modes. Both use the same patched model code
(training-order gating and MoE weighting, normalization fixes) and FP32 Mamba state.

| | default (fast) | `--exact` |
|---|---|---|
| Use for | generation, scoring and benchmarks where results only need to be close to the reference | reproducing reference logits bit for bit, regression tests, debugging numerical differences |
| `VLLM_BATCH_INVARIANT` | off | on (fixed scan reductions) |
| Training-precision arithmetic | off | on when the checkpoint enables it |
| Attention backend | chosen by vLLM | fixed (FA2 under training precision) |
| Async scheduling, compilation | vLLM defaults (on) | off |
| Bitwise guarantee | **none**: results can differ slightly with batch composition, run and engine | as described under Numerical contract |

The fast mode is the default because the measured differences to `--exact` are
small (see the validation record of this change); it is not a bitwise-reproducible
path. Use `--exact` whenever identical outputs across batches or runs matter.

## Numerical contract (`--exact`)

The exact path uses FP32 Mamba state, training-order gating and MoE weighting,
fixed scan reductions, cache-owned chunk decode, fixed TE RMSNorm, and ordinary
FA2 attention with an explicit CP reference. CP1 and CP8 Megatron references are
not bitwise interchangeable. `ceno_reference_context_length` is fixed per model,
never derived from the current batch or prefix.

Current precision support: BF16 CENO widths 512/1024, Mamba heads 16/32,
attention heads 8/16, TP=PP=1, no quantization, prefix caching or speculation.
In `--exact` mode decode CUDA Graphs remain enabled; asynchronous scheduling and
whole-model compilation are disabled. The CLI defaults to eight simultaneous requests; use `--max-num-seqs` to change it.
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

One NVIDIA H200, vLLM 0.15.1, BF16, TP=1, concurrency 1, greedy, random DNA
prompts. Output tokens per second over the whole request (prefill + decode),
excluding model loading and compilation; each point is a single measurement
(input scaling: 3 requests after 1 warm-up, 512 generated tokens; decode: 1
request from a 1024-token prompt). Values within about 10% should not be read
as real differences.

| Model | Workload | Default (fast) | `--exact` | fast / exact |
|---|---|---:|---:|---:|
| 80M | 8192-token prefix | 899 | 758 | 1.19x |
| 80M | 131072-token prefix | 745 | 260 | 2.86x |
| 80M | decode to 131072 tokens | 931 | 412 | 2.26x |
| 300M | 8192-token prefix | 813 | 638 | 1.27x |
| 300M | 131072-token prefix | 654 | 171 | 3.82x |
| 300M | decode to 131072 tokens | 834 | 292 | 2.86x |
| 600M | 8192-token prefix | 741 | 424 | 1.75x |
| 600M | 131072-token prefix | 494 | 121 | 4.07x |
| 600M | decode to 131072 tokens | 819 | 214 | 3.82x |
| 1.3B | 8192-token prefix | 575 | 261 | 2.21x |
| 1.3B | 131072-token prefix | 286 | 91.4 | 3.13x |
| 1.3B | decode to 131072 tokens | 533 | 141 | 3.79x |
| 9B | 8192-token prefix | 210 | 204 | 1.03x |
| 9B | 131072-token prefix | 74.3 | 56.9 | 1.31x |
| 9B | decode to 131072 tokens | 204 | 132 | 1.55x |

Short decodes (8192 tokens in total) can be slower in the default mode for the
smallest and largest models (80M: 669 vs 849; 9B: 196 vs 216 tokens/s).

## Accuracy of the default mode

Measured against `--exact` on the public 80M, 1.3B and 9B checkpoints:

- Next-base log-probabilities over phiX174 and four 131072-bp human windows:
  mean absolute difference at most 0.016 nats per position, top-1 agreement at
  least 99.1%, sequence NLL within 1.4e-4 nats. Individual positions can differ
  more (largest probability difference 0.57; differences above 0.1 at no more
  than 0.08% of positions).
- Zero-shot variant scoring (ClinVar chr8 test split, TraitGym Mendelian) and
  natural gene continuation: paired differences had 95% confidence intervals
  that include zero, i.e. no difference was detected at these sample sizes.
  This is not a proof of equivalence.

The default mode is approximately equal to `--exact`, not bitwise identical.
Use `--exact` when per-position scores or identical outputs across runs matter.
