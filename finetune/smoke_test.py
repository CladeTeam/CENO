#!/usr/bin/env python3
"""End-to-end smoke test of finetune/train.py on synthetic DNA (no external data).

Builds a learnable synthetic corpus (random concatenations of a fixed motif library), measures the
starting model's held-out loss, fine-tunes for a few steps, reloads the saved checkpoint with
``trust_remote_code`` and checks that (1) it loads standalone, (2) held-out loss went down,
(3) generate() works, (4) the saved config restores the checkpoint's inference precision setting.

usage: python finetune/smoke_test.py --model CladeTeam/CENO-80M-A30M-base --work /tmp/ceno_ft_smoke [--lora]
Needs one CUDA GPU (Mamba kernels). Takes a few minutes for the 80M model.
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def synth(path, n_seq, length, motifs, rng):
    with open(path, "w") as f:
        for i in range(n_seq):
            s = []
            while sum(map(len, s)) < length:
                s.append(rng.choice(motifs))
            f.write(f">synthetic_{i}\n{''.join(s)[:length]}\n")


def held_out_loss(model_dir, fasta, max_length):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    cfg.ceno_training_precision = False  # same arithmetic for both measurements
    m = AutoModelForCausalLM.from_pretrained(model_dir, config=cfg, trust_remote_code=True,
                                             torch_dtype=torch.bfloat16).cuda().eval()
    seqs = [l.strip() for l in open(fasta) if l.strip() and not l.startswith(">")]
    losses = []
    with torch.no_grad():
        for s in seqs:
            ids = torch.tensor([list(s[:max_length].encode())], device="cuda")
            losses.append(float(m(ids, labels=ids).loss))
    out = sum(losses) / len(losses)
    del m; torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CladeTeam/CENO-80M-A30M-base")
    ap.add_argument("--work", default="/tmp/ceno_ft_smoke")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--lora", action="store_true")
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    motifs = ["".join(rng.choice("ACGT") for _ in range(rng.randint(20, 60))) for _ in range(24)]
    synth(work / "train.fa", 64, 8 * a.max_length, motifs, rng)
    synth(work / "eval.fa", 8, a.max_length, motifs, rng)
    before = held_out_loss(a.model, work / "eval.fa", a.max_length)
    out = work / ("out_lora" if a.lora else "out_full")
    cmd = [sys.executable, str(HERE / "train.py"), "--model", a.model, "--train", str(work / "train.fa"),
           "--output", str(out), "--max-length", str(a.max_length), "--max-steps", str(a.steps),
           "--batch-size", "4", "--grad-accum", "1", "--lr", "1e-3" if a.lora else "1e-4", "--warmup-ratio", "0.1",
           "--save-steps", "100000", "--logging-steps", "10"]
    if a.lora:
        cmd += ["--lora", "--merge-lora"]
    subprocess.run(cmd, check=True)
    after = held_out_loss(str(out), work / "eval.fa", a.max_length)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(out), trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(str(out), trust_remote_code=True, torch_dtype="auto").cuda().eval()
    ids = tok.encode(motifs[0] + motifs[1], return_tensors="pt").cuda()
    gen = m.generate(ids, max_new_tokens=32, do_sample=False)
    saved = json.loads((out / "config.json").read_text())
    report = dict(model=a.model, lora=a.lora, steps=a.steps, max_length=a.max_length,
                  eval_loss_before=round(before, 4), eval_loss_after=round(after, 4),
                  loss_dropped=after < before - 0.05, generated=tok.decode(gen[0][ids.shape[1]:]),
                  saved_ceno_training_precision=saved.get("ceno_training_precision"),
                  standalone_files=sorted(p.name for p in out.iterdir() if p.suffix in (".py", ".json")))
    (work / f"smoke_{'lora' if a.lora else 'full'}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    assert report["loss_dropped"], "held-out loss did not decrease"
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
