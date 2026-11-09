# finetune — continued pretraining / SFT on your own DNA

A small, generic entry point for adapting a released **CENO base** checkpoint to your own
sequences with the ordinary next-token objective. It uses the Hugging Face `Trainer` and the
model code bundled with every checkpoint (`trust_remote_code=True`). No data ships with it.

## Install

Use the Hugging Face image from the repository root (`Dockerfile.huggingface`), or an equivalent
environment with PyTorch, Transformers 4.48, `mamba-ssm`, `causal-conv1d` and FlashAttention 2
(the same stack as HF inference). LoRA additionally needs `peft`.

## Data

Any of:

- FASTA: `.fa`, `.fasta`, `.fna` (optionally `.gz`), one record per sequence;
- plain text: one sequence per line;
- JSONL: `{"sequence": "ACGT..."}` per line.

Sequences are upper-cased; characters outside `--alphabet` (default `ACGTN`) become `N`.
Each sequence is cut into windows of `--max-length` tokens (one token per base) with step
`--stride` (default: no overlap). A final window shorter than `--max-length` is right-padded and
its padding is ignored by the loss; windows shorter than `--min-length` are dropped.

## Train

```bash
# one GPU, full fine-tuning
python finetune/train.py \
  --model CladeTeam/CENO-80M-A30M-base \
  --train my_sequences.fa --eval held_out.fa \
  --output runs/my-ceno-80m \
  --max-length 8192 --batch-size 1 --grad-accum 8 --lr 1e-5 --epochs 1

# several GPUs on one node (DDP)
torchrun --nproc-per-node 8 finetune/train.py --model CladeTeam/CENO-1.3B-A450M-base ...

# LoRA (adapters on the Mamba and attention projections), merged into the weights at the end
python finetune/train.py --model CladeTeam/CENO-300M-A105M-base ... --lora --merge-lora
```

Main options:

| Option | Default | Notes |
|---|---|---|
| `--max-length` | 8192 | Tokens (bases) per window. Longer windows need more memory; use `--gradient-checkpointing`. |
| `--weights-dtype` | `float32` | Parameter dtype. Compute uses bf16 autocast either way. `bfloat16` halves weight memory. |
| `--lr`, `--warmup-ratio`, `--weight-decay` | 1e-5, 0.03, 0.1 | AdamW (β = 0.9, 0.95), cosine schedule, gradient clipping 1.0. |
| `--lora` / `--merge-lora` | off | Without `--merge-lora` only the adapter is saved (load it with `peft`). LoRA usually needs a larger `--lr` than full fine-tuning (e.g. 1e-4 to 1e-3). |
| `--gradient-checkpointing` | off | Trades compute for activation memory. |

`accelerate launch` works as well as `torchrun`. Multi-node, tensor/sequence parallelism and
DeepSpeed/FSDP are not covered by this script.

## Output

`--output` becomes a complete checkpoint: weights, `config.json`, tokenizer files and the model
code copied from the starting checkpoint, so it loads exactly like a released model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("runs/my-ceno-80m", trust_remote_code=True, torch_dtype="auto").cuda().eval()
```

Training runs on the ordinary Hugging Face code path. The checkpoint's inference setting
(`ceno_training_precision` in `config.json`) is restored in the saved config, so the fine-tuned
model uses the same inference path as the model it started from.

## Check your setup

```bash
python finetune/smoke_test.py --model CladeTeam/CENO-80M-A30M-base --work /tmp/ceno_ft_smoke
```

It fine-tunes for 60 steps on generated synthetic sequences, reloads the result and checks that
held-out loss decreased and `generate()` works (about a few minutes on one GPU).

## Notes

- One CUDA GPU is required (Mamba kernels). CPU is not supported for training.
- Memory grows with model size and `--max-length`. For the larger checkpoints start with LoRA,
  `--weights-dtype bfloat16` and `--gradient-checkpointing`.
- For MoE layers, experts that receive no token in a step get no gradient; DDP therefore runs with
  `find_unused_parameters=True`.
