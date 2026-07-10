# CENO

**CENO** is a DNA foundation model family built on a Mamba / Attention / MoE hybrid backbone
(Nemotron-H architecture), trained on genomic sequence with multi-species alignment (MSA)
post-training. This repository contains the open-source **code** for three tasks:

| Part | Directory | What it does |
|------|-----------|--------------|
| 1. Model code | [`ceno_model/`](ceno_model) | HuggingFace `trust_remote_code` package for **CENO-P** (the MSA post-trained model). Loads via `from_pretrained`, supports both generation and MSA-based variant scoring. |
| 2. Variant Effect Prediction (VEP) | [`vep/`](vep) | HuggingFace-only VEP pipeline with a **TraitGym** worked example: load CENO-P → score WT vs. variant sequences (delta log-likelihood) → AUROC / AUPRC. |
| 3. Generation demo | [`generation/`](generation) | vLLM offline inference demo (generation + embedding) for the pre-trained DNA model, including the patcher that wires this repo's model code into a checkpoint dir. |

> **This repo ships code only.** Model weights live on the HuggingFace Hub — see
> [Checkpoints](#checkpoints) below. The code is checkpoint-agnostic: point each part at a
> checkpoint directory (`--model_dir`), either a Hub snapshot or a local checkout.

## Quickstart

```bash
pip install -r requirements.txt

# Part 1 — smoke test the model code (no weights needed):
python -m ceno_model.examples.load_model

# Part 2 — VEP self-test (synthetic MSA + mock scorer, no GPU / weights / data needed):
python -m vep.run_traitgym --self-test

# Part 3 — real generation needs a GPU, the pinned vLLM environment, and a
# patched base CENO checkpoint (see generation/README.md).
```

For the real TraitGym VEP run you additionally need: a CENO-P checkpoint, the human MSA zarr, and
the GRCh38 reference FASTA. See [`vep/README.md`](vep/README.md) for where each comes from.

## Architecture in one paragraph

CENO packs a 2-D multiple-sequence alignment `(L, D)` (length × depth) into a single 1-D token
sequence by **concatenating the MSA rows** `[row₁, row₂, …, row_K, target]` with no special
tokens, and passes a per-token `seq_idx` so the model knows the row boundaries. A per-layer
`intra_encoding_pattern` (`+` = isolate rows by resetting Mamba SSM state and masking
cross-segment attention; `-` = let rows fuse) alternates isolation and fusion across the layer
stack. The reference (human) sequence is the final `target` segment, so the model scores it
after recurrently consuming the whole alignment. When `seq_idx` is omitted, the model degrades
to ordinary causal-LM behavior — which is why the same package serves both generation and MSA
scoring. See `ceno_model/README.md` for the `from_pretrained` details.

## Checkpoints

Model weights are published on the HuggingFace Hub under the [`CladeTeam`](https://huggingface.co/CladeTeam)
organization — see the [CENO collection](https://huggingface.co/collections/CladeTeam/ceno-6a361ca49e030fe463f26546).
There are **15 checkpoints** across two families:

- **CENO** (base DNA foundation model, bfloat16): four sizes (80M / 300M / 600M / 1B) × three
  training-context stages (`base` / `131k` / `1m`) = 12 checkpoints, e.g.
  [`CladeTeam/CENO-300M-base`](https://huggingface.co/CladeTeam/CENO-300M-base).
- **CENO-P** (MSA post-trained, float32): three sizes (300M / 600M / 1B), e.g.
  [`CladeTeam/CENO-P-300M`](https://huggingface.co/CladeTeam/CENO-P-300M).

Each checkpoint directory loads standalone via `AutoModelForCausalLM.from_pretrained(...,
trust_remote_code=True)` — the model code is bundled in the checkpoint dir. Alternatively, drop
any checkpoint directory locally and pass its path as `--model_dir`. Its `config.json` carries
the `auto_map` pointing at this package's modules, e.g.:

```json
{
  "auto_map": {
    "AutoConfig": "configuration_ceno.CENOConfig",
    "AutoModelForCausalLM": "modeling_ceno.CENOForCausalLM",
    "AutoTokenizer": ["ceno_tokenizer.CENOCharLevelTokenizer", null]
  }
}
```

`generation/patch_vllm_for_dna.py` symlinks this repo's model code into a checkpoint dir so
`trust_remote_code=True` resolves it.

## License

Apache-2.0. The model code derives from NVIDIA's Nemotron-H HuggingFace implementation
and the tokenizer derives from Arc Institute's Evo2 CharLevelTokenizer (both Apache-2.0).
See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
