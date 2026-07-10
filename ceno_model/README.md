# ceno_model — CENO-P model code

HuggingFace `trust_remote_code`-compatible package for **CENO-P**, the MSA
(multi-species alignment) post-trained CENO model, built on the Nemotron-H
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
The checkpoint's `config.json` carries `auto_map` pointing at the modules above;
`generation/patch_vllm_for_dna.py` symlinks this package's files into the ckpt dir.

**B. As a Python package** (e.g. from the VEP adapter):
```python
from ceno_model.ceno_hf import CENOConfig
from ceno_model.ceno_hf.modeling_ceno_p import CENOPForCausalLM
cfg = CENOConfig.from_pretrained("<ckpt_dir>")
model = CENOPForCausalLM.from_pretrained("<ckpt_dir>", config=cfg, torch_dtype="bfloat16")
```

## Optional acceleration (env vars)

These are read by the VEP adapter:

- `CENO_ATTN_IMPLEMENTATION=flash_attention_2` — FlashAttention-2.
- `CENO_ENABLE_MAMBA_KERNELS=1` — Mamba CUDA fast path (needs `mamba_ssm` + `causal_conv1d`). This is required for CENO-P packed-MSA scoring because per-row state resets use the kernel path. Base CENO can fall back to PyTorch for slow single-sequence use.

## Smoke test (no weights needed)

```bash
python -m ceno_model.examples.load_model
```
