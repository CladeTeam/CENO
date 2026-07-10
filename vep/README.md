# vep — Variant Effect Prediction with CENO-P (HuggingFace)

A **pure-HuggingFace** VEP pipeline (no vLLM) that scores variants with the
MSA-enabled CENO-P model and reports AUROC / AUPRC. Ships a **TraitGym**
worked example (Mendelian traits by default; complex traits via `--task`).

The scoring signal is the **delta mean next-token log-likelihood** of the
target (reference/human) segment between the wild-type and mutant packed-MSA
inputs: a variant that lowers the model's likelihood of the reference sequence
is scored as more deleterious.

## Files

| File | Role |
|------|------|
| `base_model.py` | `BaseModel` ABC — the interface (`score_sequences`) every scorer implements. |
| `base_dataset.py` | `BaseDataset` / `TaskType` — dataset ABC (classification / regression). |
| `dataset.py` | `CENOMSADataset` — builds WT/MT packed-MSA samples from a variant CSV + human MSA zarr + GRCh38 reference FASTA. |
| `model_adapter.py` | `CENOMSAModel` — wraps the CENO-P MSA modeling path behind `BaseModel`; unpacks the packed-MSA bytes, builds the per-token `seq_idx`, and scores only the last (target) segment. |
| `evaluator.py` | `VEPEvaluator` — runs the model over unique references + variants, computes delta scores, AUROC/AUPRC/etc., writes `metrics.json` + `detailed_scores.tsv`. |
| `run_traitgym.py` | Entry point for the TraitGym example (real run + `--self-test`). |
| `data/traitgym_sample.csv` | A 24-variant balanced subset of TraitGym `mendelian_traits` for trying the real-run command. |

The bundled sample is derived from the
[`songlab-cal/TraitGym`](https://github.com/songlab-cal/TraitGym) benchmark,
which is distributed under the MIT License. Please cite the upstream TraitGym
publication when using these data.

## Packed-MSA input format

Each sample is a `bytes` blob (so PyTorch's `default_collate` leaves it alone)
encoding `(token_ids, segment_sizes)`:

```
uint16 n_segments
uint32 seg_0_len ... seg_{n-1}_len
uint32 n_token_bytes
bytes  token_bytes            # byte-level tokenizer: DNA char -> ASCII code
```

Layout: `[ctx_row_1, ctx_row_2, …, ctx_row_K, target]`. The adapter turns
`segment_sizes` into a per-token `seq_idx` (segment id) and scores only the
**last segment**. See `ceno_model/README.md` for what `seq_idx` +
`intra_encoding_pattern` do inside the model.

## Quickstart

### Self-test (no GPU / weights / real data)

Validates the whole data→eval pipeline with a synthetic MSA zarr, a synthetic
reference FASTA, and a mock scorer:

```bash
pip install -r requirements.txt
python -m vep.run_traitgym --self-test
```

Expect `SELF-TEST PASSED` with AUROC ≈ 0.5 (random mock scorer).

### Real TraitGym run

Needs: a CENO-P checkpoint, the human MSA zarr, and the GRCh38 reference FASTA.

```bash
python -m vep.run_traitgym \
    --model_dir <CENO-P checkpoint> \
    --data_csv vep/data/traitgym_sample.csv \
    --msa_zarr_path <human MSA zarr> \
    --reference_fasta <GRCh38.fa.gz>
```

`--msa_zarr_path` / `--reference_fasta` also read the `CENO_MSA_ZARR` /
`CENO_REFERENCE_FASTA` env vars. Use `--task complex_traits` for the complex-traits
split, and `--max_samples N` to subsample (balanced).

### TraitGym input CSV

A CSV with one variant per row. Recognized columns (name matching is automatic):
`chrom`/`CHROM`, `pos`/`POS`, `ref`/`REF`, `alt`/`ALT`, and a label column. For
TraitGym the label column is `label` with string values `True` (positive) /
`False` (negative). Coordinates are **GRCh38** — no liftover is applied (the
model and MSA are also hg38).

## Where the large files come from

This repo ships code only. For a real run you must supply:

- **CENO-P checkpoint** — a directory with `config.json` + `*.safetensors`
  whose `config.json` carries the `auto_map` to this repo's model modules (see
  the top-level README → *Checkpoints*).
- **Human MSA zarr** — a zarr group keyed by chromosome, each an `(L, D)`
  uint8 array whose column 0 is the reference. (Produced upstream by the MSA
  pipeline.)
- **GRCh38 reference FASTA** — e.g. `Homo_sapiens_assembly38.fasta`, gzipped ok.
- **TraitGym variant table** — the TraitGym benchmark
  (`mendelian_traits` / `complex_traits` `test.csv`), or any CSV with the
  columns above.
