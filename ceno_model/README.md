# ceno_model — CENO and CENO-P model code

HuggingFace `trust_remote_code`-compatible code for base **CENO** generation and
**CENO-P** MSA (multi-species alignment) scoring. Both use the Nemotron-H
Mamba / Attention / MoE hybrid backbone.

## Files

| File | Role |
|------|------|
| `ceno_hf/configuration_ceno.py` | `CENOConfig`. Carries `hybrid_override_pattern` (per-layer block type: `M`=Mamba, `*`=Attention, `E`=MoE, `-`=dense MLP) and the CENO addition `intra_encoding_pattern` (`+`/`-` per layer). The `intra_encoding_mask` property turns the pattern into a per-layer bool list. |
| `ceno_hf/modeling_ceno.py` | `CENOForCausalLM` — base causal-LM forward. `seq_idx` is optional; when omitted the model behaves as an ordinary causal LM. **Use this for generation / single-sequence scoring.** |
| `ceno_hf/modeling_ceno_p.py` | `CENOPForCausalLM` with the **MSA scoring path**: consumes a per-token `seq_idx` (packed MSA rows) and applies `intra_encoding_mask` per layer — `+` resets Mamba SSM state + masks cross-segment attention (isolate rows), `-` lets rows fuse. **Use this for VEP scoring** (see `vep/model_adapter.py`). |
| `ceno_hf/ceno_tokenizer.py` | `CENOCharLevelTokenizer` — byte-level tokenizer (vocab 512). DNA chars map to their ASCII code. |
| `ceno_hf/{tokenizer_config,vocab,special_tokens_map,generation_config}.json` | tokenizer / generation metadata. |

## Two ways to load

**A. As a checkpoint's remote code** (`trust_remote_code=True`):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("<ckpt_dir>", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained("<ckpt_dir>", trust_remote_code=True)
```
The checkpoint's `config.json` carries `auto_map` pointing at the modules above.
For vLLM, `generation/patch_vllm_for_dna.py` prepares a separate disposable runtime.

**B. As a Python package** (e.g. from the VEP adapter):
```python
from ceno_model.ceno_hf import CENOConfig
from ceno_model.ceno_hf.modeling_ceno_p import CENOPForCausalLM
cfg = CENOConfig.from_pretrained("<ckpt_dir>")
model = CENOPForCausalLM.from_pretrained("<ckpt_dir>", config=cfg, torch_dtype="auto")
```

## Attention implementation and long sequences

When you do not pass `attn_implementation`, base CENO checkpoints
(`CENOForCausalLM`) choose the attention implementation as follows; the choice is
logged at info level (`transformers.logging.set_verbosity_info()` to see it).

| Environment | Default attention |
|---|---|
| `flash-attn` installed, CUDA GPU with compute capability ≥ 8.0 (Ampere or newer), BF16/FP16 weights | `flash_attention_2` |
| any condition above not met (no `flash-attn`, CPU-only, older GPU, FP32) | `sdpa` |
| `attn_implementation=...` passed explicitly | the value you passed |
| CENO-P (`CENOPForCausalLM`) | unchanged (`sdpa` unless passed); its MSA segment masks need an explicit mask |

With `sdpa` (and `eager`, which also calls SDPA in this implementation), inputs
without padding use causal attention without an explicit mask.
An explicit mask is still needed for padded batches, `seq_idx` segments and cached
chunks; for sequences longer than 65536 tokens that case raises an error, because
some PyTorch builds compute SDPA with masks of more than 2**32 elements incorrectly.
**For sequences longer than 65536 tokens use `attn_implementation="flash_attention_2"`
or the vLLM entry.** `seq_idx` segment masks require `sdpa` or `eager`.

## CUDA kernels and VEP settings

These are read by the VEP adapter:

- `CENO_ATTN_IMPLEMENTATION=flash_attention_2` — FlashAttention-2.
- `CENO_ENABLE_MAMBA_KERNELS=1` — Mamba CUDA fast path (needs `mamba_ssm` + `causal_conv1d`). This is required for CENO-P packed-MSA scoring because per-row state resets use the kernel path.

The configured base CENO CUDA BF16 precision path also requires these kernels
and FlashAttention-2; it does not silently switch to the slow fallback. Base
CENO on CPU or with explicit FP32 uses the ordinary compatible path.

Install CUDA-compatible builds for the active PyTorch environment with:

```bash
pip install mamba-ssm==2.2.4 causal-conv1d==1.2.2.post1 flash-attn==2.7.3 --no-build-isolation
```

## Smoke test (no weights needed)

```bash
python -m ceno_model.examples.load_model
```

## Inference maintenance

Standalone base model repositories include `training_precision.py`, `ceno_norm.py`,
`ceno_cp_math.py` and the upstream normalization license. Imports are relative to
the model directory; HF does not require vLLM. Current base checkpoint metadata
selects the corrected CUDA BF16 inference path. Use `torch_dtype="auto"` to retain
the checkpoint dtype. The exact path is single-GPU, inference-only, unpadded,
and limited to 131072 prefill tokens; larger prefills use the vLLM entry.

CENO-P uses `modeling_ceno_p.py`, preserves `seq_idx` and MSA segment boundaries,
and retains its own configured dtype. Base generation's bitwise reference claim
must not be applied to CENO-P scoring. Model weights and token IDs are unchanged.

The reference HF environment uses PyTorch 2.7.0a0 (NVIDIA 25.03), Transformers
4.48.3, mamba-ssm 2.2.4, causal-conv1d 1.2.2.post1 and FlashAttention 2.7.3.
The cluster tests use the existing environment; no installation is needed there.
The public Dockerfile records the matching build recipe and was not rebuilt during
this maintenance. Exact numerical evidence applies to the tested environment.
Explicit `attn_implementation="eager"` is preserved for attention-map analysis.
