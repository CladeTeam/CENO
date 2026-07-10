# generation — vLLM inference demo for CENO (DNA generation)

A vLLM-based demo for **DNA sequence generation** (and embedding) with the
pre-trained CENO model. This is the fast, batched alternative to
`model.generate()` — for the MSA-based VEP scoring task use [`vep/`](../vep)
(HuggingFace forward) instead.

> vLLM is heavy and GPU-only — install the pinned integration with
> `pip install -r requirements-vllm.txt` in a CUDA container. The patcher refuses
> unvalidated vLLM versions. The FlashInfer attention backend is recommended but
> can be overridden with `VLLM_ATTENTION_BACKEND`.

## Files

| File | Role |
|------|------|
| `patch_vllm_for_dna.py` | One-time patcher: synchronizes this repo's model/tokenizer files into a **base CENO** checkpoint, persists the vLLM-supported architecture identifier, and patches the pinned container-local vLLM source. |
| `vllm_offline_infer.py` | Offline **generation** demo. |
| `vllm_offline_embed.py` | Offline **embedding** demo. |

## Quickstart (3 steps)

### 1. Prepare a checkpoint dir

Point the patcher at a CENO checkpoint directory (`config.json` + weights):

```bash
python3 generation/patch_vllm_for_dna.py /path/to/ceno_checkpoint
```

What it does (idempotent):
1. Synchronizes this release's model/tokenizer files into the checkpoint so the
   patched checkpoint remains standalone-loadable.
2. Persists `architectures: ["NemotronHForCausalLM"]` for vLLM dispatch while
   retaining CENO's `model_type` and HuggingFace `auto_map`.
3. Patches the container's vLLM source to support the Nemotron-H architecture
   (container-local; does not touch the host). vLLM's `nemotron_h.py` is vLLM's
   own file, so its class names stay `NemotronH*` — only their bodies are patched.
4. Registers `"ceno"` / `"CENOForCausalLM"` as aliases in vLLM's model registry so
   a checkpoint with `model_type: "ceno"` dispatches to that same patched code.

> **Patch before inferring.** This generation path supports base **CENO** checkpoints
> only; CENO-P MSA scoring remains HuggingFace-only. When switching checkpoints in the same container,
> re-run the patcher for the new dir (the vLLM source patch is not repeated).

By default the patcher copies from `ceno_model/ceno_hf/` in this repo.
If you keep the model code elsewhere, set `CENO_SHARED_CODE_DIR`.

### 2. Generate

```bash
python3 generation/vllm_offline_infer.py /path/to/ceno_checkpoint \
    --prompts "ATCGATCGATCG,GATTACAGATTACA" \
    --max-tokens 256 \
    --temperature 0.9
```

| Flag | Default | Meaning |
|------|---------|---------|
| `model_dir` | (required) | checkpoint directory |
| `--prompts` | built-in examples | comma-separated DNA prompts |
| `--max-tokens` | 128 | max generation length (≈ HF `max_new_tokens`) |
| `--temperature` | 0.8 | 0 = greedy, 0.8–1.0 = diverse sampling |
| `--top-p` | 0.95 | nucleus sampling threshold |
| `--top-k` | None | top-k sampling |
| `--max-model-len` | 8192 | max model context length |
| `--tp` | 1 | tensor-parallel size (multi-GPU) |

Or inline in Python:

```python
import os
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
from vllm import LLM, SamplingParams

llm = LLM(model="/path/to/ceno_checkpoint", trust_remote_code=True,
          dtype="bfloat16", max_model_len=8192, enable_prefix_caching=False)
outputs = llm.generate(["ATCGATCG", "GATTACA"],
                       SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128))
for o in outputs:
    print(o.prompt, "->", o.outputs[0].text)
```

### 3. (Optional) FP8 quantization — Hopper / Ada Lovelace GPUs

W8A8 FP8 roughly halves memory and ~1.6× throughput, with **no pre-quantized
checkpoint** needed — pass flags to use a BF16 checkpoint directly:

```bash
python3 generation/vllm_offline_infer.py /path/to/ceno_checkpoint \
    --quantization fp8 --kv-cache-dtype fp8
```

| Option | Effect | Applies to |
|--------|--------|------------|
| `--quantization fp8` | Linear-layer W8A8 | Attention proj, MoE/MLP |
| `--kv-cache-dtype fp8` | FP8 KV cache | Attention layers |
| — | Mamba SSM layers unaffected (no Linear / KV cache) | — |

> Online dynamic quantization loads the model at full precision first — make
> sure you have enough GPU memory for the full model. A100 (Ampere) supports
> only weight-only W8A16 (Marlin), **not** W8A8.
