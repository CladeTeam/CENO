#!/usr/bin/env python3
"""Continued pretraining / SFT of a CENO base checkpoint on your own DNA sequences.

Causal language modelling (next-token loss) with the Hugging Face Trainer, starting from any
CENO base checkpoint on the Hub (e.g. ``CladeTeam/CENO-80M-A30M-base``) or a local directory.

Input: FASTA (``.fa .fasta .fna``, optionally ``.gz``), plain text (one sequence per line) or
JSONL (``{"sequence": "ACGT..."}`` per line). Sequences are upper-cased, tokenized byte-wise
(the CENO tokenizer maps each character to its ASCII code) and cut into windows of
``--max-length`` tokens with ``--stride``; the last, shorter window of a record is right-padded
and its padding is excluded from the loss.

Single GPU:   python finetune/train.py --model CladeTeam/CENO-80M-A30M-base --train my.fa --output out
Multi GPU:    torchrun --nproc-per-node 8 finetune/train.py ...   (DDP; see finetune/README.md)

The output directory is a complete, ``trust_remote_code``-loadable checkpoint (weights, config,
tokenizer and the bundled model code copied from the starting checkpoint).
"""
import argparse
import gzip
import json
import math
import os
import random
import shutil
from pathlib import Path

import torch
from torch.utils.data import Dataset

CODE_SUFFIXES = (".py",)
KEEP_FILES = ("tokenizer_config.json", "special_tokens_map.json", "vocab.json", "generation_config.json",
              "LICENSE", "NOTICE")


# ----------------------------------------------------------------------------- data
def _open(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def read_sequences(path):
    """Yield (name, sequence) from FASTA, text (one sequence per line) or JSONL."""
    name = str(path).lower().removesuffix(".gz")
    with _open(path) as f:
        if name.endswith((".fa", ".fasta", ".fna")):
            header, chunks = None, []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if chunks:
                        yield header, "".join(chunks)
                    header, chunks = line[1:], []
                elif line:
                    chunks.append(line)
            if chunks:
                yield header, "".join(chunks)
        elif name.endswith(".jsonl"):
            for i, line in enumerate(f):
                if line.strip():
                    yield str(i), json.loads(line)["sequence"]
        else:
            for i, line in enumerate(f):
                if line.strip():
                    yield str(i), line.strip()


class WindowDataset(Dataset):
    """Fixed-length token windows over the input sequences."""

    def __init__(self, paths, max_length, stride, min_length, pad_id, allowed):
        self.max_length, self.pad_id = max_length, pad_id
        self.windows = []
        for p in paths:
            for _, seq in read_sequences(p):
                seq = seq.upper()
                if allowed:
                    seq = "".join(c if c in allowed else "N" for c in seq)
                ids = list(seq.encode("ascii", errors="replace"))
                for start in range(0, max(1, len(ids) - min_length + 1), stride):
                    w = ids[start:start + max_length]
                    if len(w) >= min_length:
                        self.windows.append(w)
                    if start + max_length >= len(ids):
                        break
        if not self.windows:
            raise ValueError("no training windows: check the input files and --min-length")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]
        pad = self.max_length - len(w)
        ids = torch.tensor(w + [self.pad_id] * pad, dtype=torch.long)
        labels = ids.clone()
        if pad:
            labels[-pad:] = -100
        return {"input_ids": ids, "labels": labels}


# ----------------------------------------------------------------------------- model
def resolve_dir(model):
    if Path(model).is_dir():
        return Path(model)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model, allow_patterns=["*.py", "*.json", "LICENSE", "NOTICE", "third_party/*"]))


def load_model(model, dtype, attn):
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    reference_precision = bool(getattr(config, "ceno_training_precision", False))
    # The bitwise reference-precision path is inference-only; training uses the ordinary HF path.
    config.ceno_training_precision = False
    kwargs = dict(config=config, trust_remote_code=True, torch_dtype=dtype)
    if attn:
        kwargs["attn_implementation"] = attn
    return AutoModelForCausalLM.from_pretrained(model, **kwargs), reference_precision


def add_lora(model, r, alpha, dropout, targets):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets, bias="none",
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def finalize(output, source_dir, reference_precision):
    """Make the output a self-contained trust_remote_code checkpoint."""
    output = Path(output)
    for f in source_dir.iterdir():
        if f.is_file() and (f.suffix in CODE_SUFFIXES or f.name in KEEP_FILES) and not (output / f.name).exists():
            shutil.copy2(f, output / f.name)
    if (source_dir / "third_party").is_dir() and not (output / "third_party").exists():
        shutil.copytree(source_dir / "third_party", output / "third_party")
    cfg_path = output / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        cfg["ceno_training_precision"] = reference_precision  # restore the inference default
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hub id or local CENO base checkpoint")
    ap.add_argument("--train", nargs="+", required=True, help="FASTA / text / JSONL files")
    ap.add_argument("--eval", nargs="*", default=[], help="optional held-out files (same formats)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-length", type=int, default=8192, help="tokens per training window")
    ap.add_argument("--stride", type=int, default=None, help="window step (default: --max-length)")
    ap.add_argument("--min-length", type=int, default=128, help="drop windows shorter than this")
    ap.add_argument("--alphabet", default="ACGTN", help="other characters become N; empty string keeps all")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=1, help="per device")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--weights-dtype", choices=("float32", "bfloat16"), default="float32",
                    help="parameter dtype; compute is bf16 autocast either way")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--attn-implementation", default="flash_attention_2")
    ap.add_argument("--lora", action="store_true", help="train LoRA adapters instead of all weights (needs peft)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--lora-targets", nargs="+", default=["in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"])
    ap.add_argument("--merge-lora", action="store_true", help="merge adapters into the weights before saving")
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from transformers import Trainer, TrainingArguments, set_seed
    set_seed(a.seed); random.seed(a.seed)
    source_dir = resolve_dir(a.model)
    model, reference_precision = load_model(a.model, getattr(torch, a.weights_dtype), a.attn_implementation)
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if a.lora:
        model = add_lora(model, a.lora_r, a.lora_alpha, a.lora_dropout, a.lora_targets)
    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else 1
    stride = a.stride or a.max_length
    train = WindowDataset(a.train, a.max_length, stride, a.min_length, pad_id, set(a.alphabet))
    evalset = WindowDataset(a.eval, a.max_length, a.max_length, a.min_length, pad_id, set(a.alphabet)) if a.eval else None
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"train windows: {len(train)}  eval windows: {len(evalset) if evalset else 0}  "
              f"tokens/window: {a.max_length}", flush=True)

    args = TrainingArguments(
        output_dir=a.output, num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, warmup_ratio=a.warmup_ratio,
        weight_decay=a.weight_decay, adam_beta1=0.9, adam_beta2=0.95, max_grad_norm=1.0,
        lr_scheduler_type="cosine", bf16=True, logging_steps=a.logging_steps,
        save_steps=a.save_steps, save_total_limit=2, eval_strategy="steps" if evalset else "no",
        eval_steps=a.eval_steps if evalset else None, report_to="none", seed=a.seed,
        dataloader_num_workers=2, remove_unused_columns=False,
        ddp_find_unused_parameters=True,  # MoE: experts that receive no token in a step get no gradient
        gradient_checkpointing=a.gradient_checkpointing,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=evalset)
    trainer.train()
    if a.lora and a.merge_lora:
        trainer.model = trainer.model.merge_and_unload()
        trainer.model.save_pretrained(a.output, safe_serialization=True)
    else:
        trainer.save_model(a.output)
    if trainer.is_world_process_zero():
        if not a.lora or a.merge_lora:
            finalize(a.output, source_dir, reference_precision)
        metrics = {k: v for k, v in trainer.state.log_history[-1].items()}
        (Path(a.output) / "train_summary.json").write_text(json.dumps(
            dict(base_model=a.model, windows=len(train), max_length=a.max_length, stride=stride,
                 lora=a.lora, merged=a.merge_lora, final=metrics,
                 log=trainer.state.log_history), indent=1))
        if evalset:
            ev = trainer.evaluate()
            print(f"eval loss {ev['eval_loss']:.4f}  perplexity {math.exp(ev['eval_loss']):.3f}", flush=True)


if __name__ == "__main__":
    main()
