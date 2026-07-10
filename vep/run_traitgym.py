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

"""TraitGym Variant Effect Prediction with CENO-P (HuggingFace).

Loads the MSA-enabled CENO-P checkpoint, builds TraitGym variant samples from a
CSV (+ human MSA zarr + GRCh38 reference FASTA), scores WT vs. variant with
delta mean-log-likelihood, and reports AUROC / AUPRC.

TraitGym coordinates are already GRCh38, so **no liftover** is needed. The default
task is ``mendelian_traits``; ``complex_traits`` is also supported via ``--task``.

Run a real evaluation:
    python -m vep.run_traitgym \\
        --model_dir <CENO-P checkpoint> \\
        --data_csv vep/data/traitgym_sample.csv \\
        --msa_zarr_path <human MSA zarr> \\
        --reference_fasta <GRCh38.fa.gz>

Self-test (no GPU / weights / real data — synthetic zarr + fasta + mock scorer):
    python -m vep.run_traitgym --self-test
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

# Make the CENO repo root importable (for `ceno_model` / `vep` packages).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vep.dataset import CENOMSADataset
from vep.evaluator import VEPEvaluator
from vep.model_adapter import CENOMSAModel

# MSA window length the CENO-P model was trained with: 128 tokens per MSA row,
# with up to 90 context rows plus the target row -> 128 * 91 = 11648 tokens per sample.
# (8192 is the single-sequence context of the CENO *base* pretrain model, NOT
# the MSA window — do not use it for CENO-P VEP scoring.)
DEFAULT_MSA_WINDOW = 128

# Default locations of the large external files. Override via CLI / env.
DEFAULT_DATA_CSV = os.path.join(os.path.dirname(__file__), "data", "traitgym_sample.csv")
DEFAULT_MSA_ZARR = os.environ.get("CENO_MSA_ZARR", "")
DEFAULT_REFERENCE_FASTA = os.environ.get("CENO_REFERENCE_FASTA", "")

# TraitGym label convention: column `label`, values "True" (positive) / "False".
TRAITGYM_LABEL_COL = "label"
TRAITGYM_POSITIVE = {"true"}
TRAITGYM_NEGATIVE = {"false"}


# --------------------------------------------------------------------------- #
# Column resolution helpers (from the internal run_all.py)
# --------------------------------------------------------------------------- #
def _pick_column(df: pd.DataFrame, candidates: list, label: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"Missing required {label} column; tried: {candidates}")


# --------------------------------------------------------------------------- #
# Mock scorer (for --self-test only)
# --------------------------------------------------------------------------- #
class _MockScorer:
    """Returns deterministic pseudo-scores so the evaluator pipeline runs end-to-end
    without a GPU or checkpoint. NOT a real model."""

    def __init__(self, seed: int = 0):
        self.model_name = "mock-scorer"
        self.model_path = "<none>"
        self._rng = np.random.default_rng(seed)

    def score_sequences(self, sequences, batch_size: int = 256):
        return [float(self._rng.standard_normal()) for _ in sequences]

    def get_embedding(self, sequences, layer_name, batch_size=64, pool="mean", return_numpy=True):
        return np.zeros((len(sequences), 1024), dtype=np.float32)


# --------------------------------------------------------------------------- #
# Self-test: synthetic MSA zarr + reference FASTA + mock scorer
# --------------------------------------------------------------------------- #
def _build_synthetic_msa_zarr(zarr_path: str, chrom: str = "1", length: int = 4096, depth: int = 6):
    """Write a small synthetic (L, D) uint8 MSA zarr: column 0 = reference,
    other columns = near-identical (occasional substitution) to mimic homologs.

    The deterministic reference makes every self-test variant verifiable."""
    import gzip

    import zarr
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    rng = np.random.default_rng(42)
    bases = np.frombuffer(b"ACGT", dtype=np.uint8)
    ref = np.full(length, ord("A"), dtype=np.uint8)
    arr_data = np.empty((length, depth), dtype=np.uint8)
    arr_data[:, 0] = ref
    for d in range(1, depth):
        col = ref.copy()
        # introduce ~1% substitutions to make non-target rows non-gap, non-identical
        mut_mask = rng.random(length) < 0.01
        col[mut_mask] = bases[rng.integers(0, 4, size=mut_mask.sum())]
        arr_data[:, d] = col

    root = zarr.open_group(zarr_path, mode="w")
    root.create_dataset(name=chrom, data=arr_data, chunks=(4096, depth), dtype="u1")

    # A matching reference FASTA (the same ref sequence) in the same dir.
    fasta_path = zarr_path + ".fa.gz"
    seq_str = "".join(chr(b) for b in ref)
    with gzip.open(fasta_path, "wt") as handle:
        SeqIO.write([SeqRecord(Seq(seq_str), id=chrom, description="")], handle, "fasta")
    return fasta_path


def run_self_test(args):
    print("=" * 64)
    print("SELF-TEST: synthetic MSA zarr + reference FASTA + mock scorer")
    print("(no GPU / checkpoint / real data needed — validates the data→eval pipeline)")
    print("=" * 64)

    out_dir = args.output_dir or os.path.join(tempfile.gettempdir(), "ceno_traitgym_selftest")
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        # A small deterministic synthetic chromosome keeps this test lightweight.
        zarr_path = os.path.join(tmp, "mock_msa.zarr")
        fasta_path = _build_synthetic_msa_zarr(zarr_path, chrom="1", length=4096, depth=6)

        # Build a tiny TraitGym-like CSV whose reference alleles match the synthetic MSA.
        n_pos, n_neg = 12, 12
        rows = []
        base_pos = 1024
        for i in range(n_pos):
            rows.append({"chrom": "1", "pos": base_pos + 1000 * i,
                         "ref": "A", "alt": "C", "label": "True"})
        for i in range(n_neg):
            rows.append({"chrom": "1", "pos": 2500 + 100 * i,
                         "ref": "A", "alt": "G", "label": "False"})
        df = pd.DataFrame(rows)

        chrom_col = "chrom"
        pos_col = "pos"
        ref_col = "ref"
        alt_col = "alt"

        ds = CENOMSADataset(
            data_path="<selftest>",
            data_df=df,
            reference_fasta_path=fasta_path,
            msa_zarr_path=zarr_path,
            window_size=args.window_size or DEFAULT_MSA_WINDOW,
            chrom_col=chrom_col,
            pos_col=pos_col,
            ref_col=ref_col,
            alt_col=alt_col,
            label_col=TRAITGYM_LABEL_COL,
            positive_labels=TRAITGYM_POSITIVE,
            negative_labels=TRAITGYM_NEGATIVE,
            max_context=6,
        )
        print(f"[INFO] dataset size = {len(ds)}, unique refs = {len(ds.get_unique_reference_sequences())}")
        assert len(ds) == n_pos + n_neg

        model = _MockScorer()
        evaluator = VEPEvaluator(
            model_batch_size=16,
            dataloader_batch_size=64,
            task_type=2,
            save_detailed=True,
            num_workers=0,
        )
        results = evaluator.evaluate(model=model, dataset=ds, output_dir=out_dir, progress_bar=True)

    metrics = results["metrics"]
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path) as f:
        saved = json.load(f)
    assert "auroc" in saved["metrics"], "metrics.json missing 'auroc'"

    print("\n" + "=" * 64)
    print("SELF-TEST PASSED")
    print(f"  n_samples   = {metrics['n_samples']} (expected {n_pos + n_neg})")
    print(f"  auroc       = {metrics.get('auroc', float('nan')):.4f}  (random mock scorer -> ~0.5)")
    print(f"  metrics.json = {metrics_path}")
    print("=" * 64)
    return results


# --------------------------------------------------------------------------- #
# Real TraitGym run
# --------------------------------------------------------------------------- #
def run_traitgym(args):
    if not args.model_dir:
        raise ValueError("--model_dir is required for a real run (or use --self-test).")
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"--model_dir not found: {args.model_dir}")

    if args.data_csv:
        data_csv = args.data_csv
    elif args.task == "mendelian_traits":
        data_csv = DEFAULT_DATA_CSV
    else:
        raise ValueError(
            "--task complex_traits requires --data_csv because this release ships only "
            "the Mendelian TraitGym sample."
        )
    msa_zarr = args.msa_zarr_path or DEFAULT_MSA_ZARR
    fasta = args.reference_fasta or DEFAULT_REFERENCE_FASTA
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"--data_csv not found: {data_csv}")
    if not msa_zarr or not os.path.exists(msa_zarr):
        raise FileNotFoundError(
            "--msa_zarr_path not found. Set --msa_zarr_path or $CENO_MSA_ZARR.")
    if not fasta or not os.path.exists(fasta):
        raise FileNotFoundError(
            "--reference_fasta not found. Set --reference_fasta or $CENO_REFERENCE_FASTA.")

    out_dir = args.output_dir or os.path.join(_REPO_ROOT, "results", "traitgym", args.task, args.model_tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] model_dir   = {args.model_dir}")
    print(f"[INFO] task        = {args.task}")
    print(f"[INFO] data_csv    = {data_csv}")
    print(f"[INFO] msa_zarr    = {msa_zarr}")
    print(f"[INFO] fasta       = {fasta}")
    print(f"[INFO] output_dir  = {out_dir}")
    print(f"[INFO] window_size = {args.window_size or DEFAULT_MSA_WINDOW}")
    print(f"[INFO] max_context = {args.max_context}")

    df = pd.read_csv(data_csv)
    if args.max_samples and args.max_samples > 0:
        # balanced subsample
        pos = df[df[TRAITGYM_LABEL_COL].astype(str).str.lower().isin(TRAITGYM_POSITIVE)]
        neg = df[df[TRAITGYM_LABEL_COL].astype(str).str.lower().isin(TRAITGYM_NEGATIVE)]
        n_each = args.max_samples // 2
        df = pd.concat([
            pos.head(n_each) if len(pos) >= n_each else pos,
            neg.head(n_each) if len(neg) >= n_each else neg,
        ]).reset_index(drop=True)
        print(f"[INFO] subsampled to {len(df)} variants (max_samples={args.max_samples})")

    chrom_col = _pick_column(df, ["chrom", "CHROM", "chr", "CHR"], "chrom")
    pos_col = _pick_column(df, ["pos", "POS", "position", "POSITION"], "pos")
    ref_col = _pick_column(df, ["ref", "REF"], "ref")
    alt_col = _pick_column(df, ["alt", "ALT"], "alt")

    # No liftover: TraitGym coordinates are GRCh38 (the model/MSA coordinate space).

    ds = CENOMSADataset(
        data_path=data_csv,
        data_df=df,
        reference_fasta_path=fasta,
        msa_zarr_path=msa_zarr,
        window_size=args.window_size or DEFAULT_MSA_WINDOW,
        chrom_col=chrom_col,
        pos_col=pos_col,
        ref_col=ref_col,
        alt_col=alt_col,
        label_col=TRAITGYM_LABEL_COL,
        positive_labels=TRAITGYM_POSITIVE,
        negative_labels=TRAITGYM_NEGATIVE,
        max_context=args.max_context,
    )
    print(f"[INFO] dataset size = {len(ds)}, unique refs = {len(ds.get_unique_reference_sequences())}")

    model = CENOMSAModel(model_name=args.model_tag, model_path=args.model_dir)
    evaluator = VEPEvaluator(
        model_batch_size=args.model_batch_size,
        dataloader_batch_size=1024,
        task_type=2,
        save_detailed=True,
        num_workers=0,
    )
    results = evaluator.evaluate(model=model, dataset=ds, output_dir=out_dir, progress_bar=True)

    metrics = results["metrics"]
    print("\n" + "=" * 64)
    print(f"TRAITGYM VEP RESULTS ({args.task})")
    print(f"  n_samples = {metrics['n_samples']}")
    print(f"  AUROC     = {metrics.get('auroc', float('nan')):.4f}")
    print(f"  AUPRC     = {metrics.get('auprc', float('nan')):.4f}")
    print(f"  results   = {out_dir}")
    print("=" * 64)
    return results


def main():
    parser = argparse.ArgumentParser(description="TraitGym VEP with CENO-P (HuggingFace).")
    parser.add_argument("--model_dir", default=None, help="CENO-P checkpoint directory.")
    parser.add_argument("--model_tag", default="CENO-P", help="Tag for result logging.")
    parser.add_argument("--task", default="mendelian_traits",
                        choices=["mendelian_traits", "complex_traits"],
                        help="TraitGym task (selects the default data CSV naming).")
    parser.add_argument("--data_csv", default=None, help=f"TraitGym CSV (default {DEFAULT_DATA_CSV}).")
    parser.add_argument("--msa_zarr_path", default=None, help="Human MSA zarr (or $CENO_MSA_ZARR).")
    parser.add_argument("--reference_fasta", default=None, help="GRCh38 reference FASTA (or $CENO_REFERENCE_FASTA).")
    parser.add_argument("--window_size", type=int, default=DEFAULT_MSA_WINDOW)
    parser.add_argument("--max_context", type=int, default=90,
                        help="Max MSA context rows packed before the target segment. "
                             "Lower this to bound the packed sequence length (and the "
                             "attention memory, which is O((max_context*window_size)^2)).")
    parser.add_argument("--max_samples", type=int, default=0, help="Subsample N variants (balanced). 0=all.")
    parser.add_argument("--model_batch_size", type=int, default=1)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--self-test", action="store_true",
                        help="Run the synthetic-data + mock-scorer pipeline (no GPU/weights/data).")
    args = parser.parse_args()

    if args.self_test:
        run_self_test(args)
    else:
        run_traitgym(args)


if __name__ == "__main__":
    main()
