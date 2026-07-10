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

import gzip
import os
import struct
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import zarr
from Bio import SeqIO
from zarr.storage import LRUStoreCache

from .base_dataset import BaseDataset, TaskType


PackedSeq = bytes


def _pack_token_ids_and_segments(token_ids: List[int], segment_sizes: List[int]) -> PackedSeq:
    """
    Pack (token_ids, segment_sizes) into bytes so that PyTorch default_collate keeps it as a list of bytes.

    Format (little-endian):
      uint16  n_segments
      uint32  seg_0_len
      ...
      uint32  seg_{n-1}_len
      uint32  n_token_bytes
      bytes   token_bytes (length n_token_bytes)
    """
    segs = [int(x) for x in segment_sizes if int(x) >= 0]
    nseg = len(segs)
    if nseg > 65535:
        raise ValueError(f"Too many segments: {nseg}")

    token_arr = np.asarray(token_ids, dtype=np.uint8)
    token_bytes = token_arr.tobytes()

    header = struct.pack("<H", nseg) + struct.pack(f"<{nseg}I", *segs) + struct.pack("<I", len(token_bytes))
    return header + token_bytes


def _msa_array_to_sequences(msa_slice: np.ndarray) -> List[str]:
    if msa_slice.ndim != 2:
        raise ValueError(f"Expect 2D (L, D) array, got shape={msa_slice.shape}")

    if msa_slice.dtype.kind in ("S", "a"):
        chars = msa_slice.astype("U1")
    elif msa_slice.dtype == np.uint8:
        chars = msa_slice.view("S1").astype("U1")
    else:
        chars = msa_slice.astype("U1")

    L, D = chars.shape
    return ["".join(chars[:, j].tolist()) for j in range(D)]


def _is_all_gap_like(s: str) -> bool:
    if not s:
        return True
    if set(s) <= {"-", "N", "n"}:
        return True
    if all(ch == "-" for ch in s):
        return True
    return False


class CENOMSADataset(BaseDataset):
    """
    CENO MSA(zarr) dataset for packed context-plus-target VEP inputs.

    输出（可直接用于 VEPEvaluator）：
      __getitem__(idx) -> (wt_packed: bytes, mt_packed: bytes, label: int)

    其中 packed bytes 可被模型 wrapper 解包为：
      (token_ids: List[int], segment_sizes: List[int])

    构造逻辑：
    - 给定变异 (chrom, pos, ref, alt, strand) 与 window_size
    - 从 zarr 读取 (L, D) 的 MSA 切片（L≈window_size）
    - 将除 target_col 外的其余序列作为上下文段（可限深 max_context）
    - 将 target_col 的序列作为最后一段（参考/目标序列）
    - mt 样本只改最后一段：在目标序列窗口中应用 ref->alt（保持总长度为 window_size）
    - 模型 scoring 应只算最后一段（目标序列）的 next-token log-likelihood
    """

    def __init__(
        self,
        data_path: str,
        reference_fasta_path: str,
        msa_zarr_path: str,
        window_size: int = 128,
        chrom_col: str = "chrom",
        pos_col: str = "pos",
        ref_col: str = "ref",
        alt_col: str = "alt",
        label_col: str = "consequence",
        strand_col: Optional[str] = "strand",
        gene_col: Optional[str] = "gene",
        id_col: Optional[str] = None,
        distance_to_nearest_tss_col: Optional[str] = None,
        positive_labels: Optional[Set[str]] = None,
        negative_labels: Optional[Set[str]] = None,
        distance_thresholds: Optional[List[float]] = None,
        pad_base: str = "N",
        skip_on_mismatch: bool = False,
        reverse_complement_on_minus_strand: bool = True,
        target_col: int = 0,
        max_context: Optional[int] = 90,
        remove_all_gap: bool = True,
        zarr_consolidated: bool = False,
        zarr_cache_size_mb: int = 256,
        data_df: Optional[pd.DataFrame] = None,
    ):
        super().__init__(data_path, TaskType.CLASSIFICATION, window_size)

        self.reference_fasta_path = reference_fasta_path
        self.msa_zarr_path = msa_zarr_path
        self.chrom_col = chrom_col
        self.pos_col = pos_col
        self.ref_col = ref_col
        self.alt_col = alt_col
        self.label_col = label_col
        self.strand_col = strand_col
        self.gene_col = gene_col
        self.id_col = id_col
        self.distance_to_nearest_tss_col = distance_to_nearest_tss_col

        self.pad_base = (pad_base or "N").upper()
        self.skip_on_mismatch = bool(skip_on_mismatch)
        self.reverse_complement_on_minus_strand = bool(reverse_complement_on_minus_strand)

        self.target_col = int(target_col)
        self.max_context = (int(max_context) if max_context is not None else None)
        self.remove_all_gap = bool(remove_all_gap)
        self.zarr_consolidated = bool(zarr_consolidated)
        self.zarr_cache_size_bytes = int(max(0, zarr_cache_size_mb)) * 1024 * 1024
        self._inmem_df = data_df.copy() if data_df is not None else None

        self.positive_labels = {s.lower() for s in (positive_labels or {"over", "under"})}
        self.negative_labels = {s.lower() for s in (negative_labels or {"none"})}
        thresholds = distance_thresholds if distance_thresholds is not None else [1000, 10000, 100000]
        self.distance_thresholds = sorted([float(t) for t in thresholds])

        self._load_df()
        self._load_reference_fasta()
        self._process_labels()
        self._init_zarr()
        self._build_all_samples()

    def _load_df(self):
        if self._inmem_df is not None:
            self.df = self._inmem_df.reset_index(drop=True)
        else:
            self.df = pd.read_csv(self.data_path)
        required = [self.chrom_col, self.pos_col, self.ref_col, self.alt_col, self.label_col]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV 缺少必要列: {missing}")

        self.df[self.chrom_col] = self.df[self.chrom_col].astype(str).str.strip()
        self.df[self.pos_col] = self.df[self.pos_col].astype(int)
        self.df[self.ref_col] = self.df[self.ref_col].astype(str).str.upper().str.replace("-", "", regex=False)
        self.df[self.alt_col] = self.df[self.alt_col].astype(str).str.upper().str.replace("-", "", regex=False)

        if self.strand_col and self.strand_col in self.df.columns:
            self.df[self.strand_col] = self.df[self.strand_col].apply(self._normalize_strand)
        else:
            self.df[self.strand_col] = 1
        if self.distance_to_nearest_tss_col and self.distance_to_nearest_tss_col in self.df.columns:
            self._process_distance_to_tss()

    def _load_reference_fasta(self):
        self.chrom_to_seq: Dict[str, str] = {}
        opener = gzip.open if self.reference_fasta_path.endswith(".gz") else open
        with opener(self.reference_fasta_path, "rt") as handle:
            for rec in SeqIO.parse(handle, "fasta"):
                self.chrom_to_seq[str(rec.id)] = str(rec.seq).upper()
        if not self.chrom_to_seq:
            raise ValueError("未能从参考 FASTA 读取到任何序列记录。")

        self._fasta_ids = set(self.chrom_to_seq.keys())
        self._fasta_has_chr_prefix = any(k.startswith("chr") for k in self._fasta_ids)

    def _process_labels(self):
        labels = self.df[self.label_col].astype(str).str.strip().str.lower()
        known_pos = labels.isin(self.positive_labels)
        known_neg = labels.isin(self.negative_labels)
        known = known_pos | known_neg

        unknown_mask = ~known
        if int(unknown_mask.sum()) > 0:
            self.df = self.df.loc[known].copy().reset_index(drop=True)
            labels = labels.loc[known]
            known_pos = known_pos.loc[known]

        if self.df.empty:
            raise ValueError("过滤未知标签后数据为空，请检查 label_col/positive_labels/negative_labels。")
        self.df["binary_label"] = np.where(known_pos, 1, 0).astype(int)

    def _process_distance_to_tss(self):
        if not self.distance_to_nearest_tss_col or self.distance_to_nearest_tss_col not in self.df.columns:
            return

        distances = pd.to_numeric(self.df[self.distance_to_nearest_tss_col], errors="coerce")
        thresholds = self.distance_thresholds

        if not thresholds:
            def categorize_distance(dist):
                if pd.isna(dist):
                    return None
                return "all"
        else:
            def format_label(start, end):
                if start is None:
                    return f"<{self._format_threshold(end)}"
                if end is None:
                    return f">{self._format_threshold(start)}"
                return f"{self._format_threshold(start)}-{self._format_threshold(end)}"

            def categorize_distance(dist):
                if pd.isna(dist):
                    return None
                dist = abs(float(dist))
                for i, threshold in enumerate(thresholds):
                    if dist < threshold:
                        prev_threshold = thresholds[i - 1] if i > 0 else None
                        return format_label(prev_threshold, threshold)
                return format_label(thresholds[-1], None)

        self.df["tss_distance_group"] = distances.apply(categorize_distance)

    def _format_threshold(self, value: float) -> str:
        if value is None:
            return ""
        if value < 1000:
            return f"{int(value)}"
        if value < 1000000:
            return f"{int(value / 1000)}k"
        return f"{int(value / 1000000)}M"

    def _init_zarr(self):
        if not os.path.exists(self.msa_zarr_path):
            raise FileNotFoundError(f"msa_zarr_path not found: {self.msa_zarr_path}")

        try:
            store = zarr.DirectoryStore(self.msa_zarr_path)
        except Exception:
            store = None

        if store is not None and self.zarr_cache_size_bytes > 0:
            store = LRUStoreCache(store, max_size=self.zarr_cache_size_bytes)

        # Default to `zarr.open` (no consolidated metadata required). If `zarr_consolidated=True`,
        # try consolidated first but fall back gracefully when `.zmetadata` is missing.
        if store is not None:
            if self.zarr_consolidated:
                try:
                    self._zgroup = zarr.open_consolidated(store=store, mode="r")
                except Exception:
                    self._zgroup = zarr.open(store=store, mode="r")
            else:
                self._zgroup = zarr.open(store=store, mode="r")
        else:
            if self.zarr_consolidated:
                try:
                    self._zgroup = zarr.open_consolidated(self.msa_zarr_path, mode="r")
                except Exception:
                    self._zgroup = zarr.open(self.msa_zarr_path, mode="r")
            else:
                self._zgroup = zarr.open(self.msa_zarr_path, mode="r")

        self._zarr_arrays: Dict[str, zarr.Array] = {}

    def _get_zarr_array(self, chrom: str) -> zarr.Array:
        if chrom in self._zarr_arrays:
            return self._zarr_arrays[chrom]
        if chrom in self._zgroup:
            arr = self._zgroup[chrom]
            self._zarr_arrays[chrom] = arr
            return arr
        # Try adding/removing chr prefix
        if chrom.startswith("chr"):
            cand = chrom[3:]
        else:
            cand = f"chr{chrom}"
        if cand in self._zgroup:
            arr = self._zgroup[cand]
            self._zarr_arrays[chrom] = arr
            return arr
        raise KeyError(f"Chromosome '{chrom}' not found in zarr group.")

    def _encode_to_byte_ids(self, seq: str) -> List[int]:
        if not seq:
            return []
        b = np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)
        return b.astype(np.int64).tolist()

    def _normalize_chrom(self, chrom: str) -> str:
        c = chrom.strip()
        if c in self._fasta_ids:
            return c
        if self._fasta_has_chr_prefix:
            cand = c if c.startswith("chr") else f"chr{c}"
        else:
            cand = c[3:] if c.lower().startswith("chr") else c
        if cand in self._fasta_ids:
            return cand
        return c

    def _normalize_strand(self, v: Union[int, str]) -> int:
        s = str(v).strip().lower()
        if s in {"1", "+", "plus", "+1"}:
            return 1
        if s in {"-1", "-", "minus"}:
            return -1
        return 1

    def _revcomp(self, seq: str) -> str:
        comp = {
            "A": "T", "T": "A", "C": "G", "G": "C", "N": "N",
            "R": "Y", "Y": "R", "S": "S", "W": "W",
            "K": "M", "M": "K", "B": "V", "V": "B",
            "D": "H", "H": "D",
            "-": "-",
            "?": "?",
        }
        return "".join(comp.get(b, b) for b in reversed(seq.upper()))

    def _apply_variant_fixed_window(self, wt: str, local_p: int, ref: str, alt: str) -> str:
        ref_len = len(ref)
        if ref_len > 0:
            ref_sub = wt[local_p: local_p + ref_len]
            if ref_sub != ref:
                raise AssertionError(f"Reference mismatch: '{ref_sub}' != '{ref}'")

        mt = wt[:local_p] + alt + wt[local_p + ref_len:]

        if len(mt) > self.window_size:
            extra = len(mt) - self.window_size
            cut_left = extra // 2
            cut_right = extra - cut_left
            mt = mt[cut_left: len(mt) - cut_right]
        elif len(mt) < self.window_size:
            pad = self.window_size - len(mt)
            pad_left = pad // 2
            pad_right = pad - pad_left
            mt = (self.pad_base * pad_left) + mt + (self.pad_base * pad_right)
        return mt

    def _fetch_msa_window(self, chrom: str, pos1: int) -> Tuple[np.ndarray, int, int]:
        arr = self._get_zarr_array(chrom)
        L = int(arr.shape[0])
        half = self.window_size // 2
        start0 = (pos1 - 1) - half
        end0 = start0 + self.window_size
        start = max(0, start0)
        end = min(L, end0)
        pad_left = int(start - start0)
        pad_right = int(end0 - end)
        return np.asarray(arr[start:end]), pad_left, pad_right

    def _build_one_sample(self, chrom: str, pos1: int, ref: str, alt: str, strand: int) -> Tuple[PackedSeq, PackedSeq]:
        half = self.window_size // 2
        local_p = half  # keep variant centered even when near chrom ends

        msa_ld, pad_left, pad_right = self._fetch_msa_window(chrom, pos1)
        seqs = _msa_array_to_sequences(msa_ld)
        if not seqs:
            raise ValueError("Empty MSA slice")

        # Pad left/right if out-of-bounds (msa slice shorter than window_size)
        if pad_left > 0 or pad_right > 0:
            seqs = [(self.pad_base * pad_left) + s + (self.pad_base * pad_right) for s in seqs]
        # Ensure equal length
        if len(seqs[0]) != self.window_size:
            # Extremely defensive: if zarr slice length doesn't match expected, center-pad to window_size
            cur_len = len(seqs[0])
            if cur_len < self.window_size:
                extra = self.window_size - cur_len
                pl = extra // 2
                pr = extra - pl
                seqs = [(self.pad_base * pl) + s + (self.pad_base * pr) for s in seqs]
            else:
                seqs = [s[: self.window_size] for s in seqs]

        if self.target_col < 0 or self.target_col >= len(seqs):
            raise IndexError(f"target_col {self.target_col} out of range for MSA depth {len(seqs)}")

        target_wt = seqs[self.target_col]

        target_mt = self._apply_variant_fixed_window(target_wt, local_p, ref, alt)

        ctx = [seqs[i] for i in range(len(seqs)) if i != self.target_col]
        if self.remove_all_gap:
            ctx = [s for s in ctx if not _is_all_gap_like(s)]
        if self.max_context is not None and self.max_context > 0:
            ctx = ctx[: self.max_context]

        if self.reverse_complement_on_minus_strand and strand == -1:
            ctx = [self._revcomp(s) for s in ctx]
            target_wt = self._revcomp(target_wt)
            target_mt = self._revcomp(target_mt)

        # Layout: ctx1 ctx2 ... ctxK target
        def _linearize(ctx_list: List[str], target: str) -> Tuple[List[int], List[int]]:
            ids_concat: List[int] = []
            seg_sizes: List[int] = []
            for s in ctx_list:
                ids = self._encode_to_byte_ids(s)
                ids_concat.extend(ids)
                seg_sizes.append(len(ids))
            tgt_ids = self._encode_to_byte_ids(target)
            ids_concat.extend(tgt_ids)
            seg_sizes.append(len(tgt_ids))
            return ids_concat, seg_sizes

        wt_ids, wt_segs = _linearize(ctx, target_wt)
        mt_ids, mt_segs = _linearize(ctx, target_mt)

        return _pack_token_ids_and_segments(wt_ids, wt_segs), _pack_token_ids_and_segments(mt_ids, mt_segs)

    def _build_all_samples(self):
        self.ref_packed_unique: List[PackedSeq] = []
        self._ref_packed_to_id: Dict[PackedSeq, int] = {}
        self.ref_unique_idx: List[int] = []
        self.var_packed: List[PackedSeq] = []

        keep_mask = np.ones(len(self.df), dtype=bool)

        for i, row in self.df.iterrows():
            chrom = self._normalize_chrom(str(row[self.chrom_col]))
            pos1 = int(row[self.pos_col])
            ref = str(row[self.ref_col])
            alt = str(row[self.alt_col])
            strand = int(row[self.strand_col]) if self.strand_col in row else 1

            try:
                wt_packed, mt_packed = self._build_one_sample(chrom, pos1, ref, alt, strand)
            except Exception as e:
                if self.skip_on_mismatch:
                    keep_mask[i] = False
                    continue
                raise RuntimeError(
                    f"Failed to build variant {chrom}:{pos1} {ref}>{alt}. "
                    "Check that the MSA, reference FASTA, and variant table share a genome build, "
                    "or explicitly set skip_on_mismatch=True."
                ) from e

            if wt_packed not in self._ref_packed_to_id:
                self._ref_packed_to_id[wt_packed] = len(self.ref_packed_unique)
                self.ref_packed_unique.append(wt_packed)
            self.ref_unique_idx.append(self._ref_packed_to_id[wt_packed])
            self.var_packed.append(mt_packed)

        self.ref_unique_idx = np.asarray(self.ref_unique_idx, dtype=np.int32)
        if self.skip_on_mismatch and not keep_mask.all():
            self.df = self.df.loc[keep_mask].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.var_packed)

    def __getitem__(self, idx: int):
        ref_idx = int(self.ref_unique_idx[idx])
        wt_packed = self.ref_packed_unique[ref_idx]
        mt_packed = self.var_packed[idx]
        label = int(self.df.iloc[idx]["binary_label"])
        return wt_packed, mt_packed, label

    def get_unique_reference_sequences(self) -> list:
        return self.ref_packed_unique

    def get_variant_info(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        info = {
            "chrom": row[self.chrom_col],
            "position": int(row[self.pos_col]),
            "ref": row[self.ref_col],
            "alt": row[self.alt_col],
            "strand": int(row[self.strand_col]) if self.strand_col in row else 1,
            "label_text": row[self.label_col],
            "label_bin": int(row["binary_label"]),
        }
        if self.gene_col and self.gene_col in self.df.columns:
            info["gene"] = row[self.gene_col]
        if self.id_col and self.id_col in self.df.columns:
            info["variant_id"] = row[self.id_col]
        if self.distance_to_nearest_tss_col and self.distance_to_nearest_tss_col in self.df.columns:
            dist = row[self.distance_to_nearest_tss_col]
            info["distance_to_nearest_tss"] = float(dist) if pd.notna(dist) else None
            if "tss_distance_group" in self.df.columns:
                info["tss_distance_group"] = row["tss_distance_group"]
        return info

    def get_dataset_info(self) -> dict:
        base = super().get_dataset_info()
        counts = self.df["binary_label"].value_counts().to_dict()
        base.update(
            {
                "num_samples": len(self),
                "num_unique_ref": len(self.ref_packed_unique),
                "window_size": self.window_size,
                "label_distribution": counts,
                "positive_ratio": float(self.df["binary_label"].mean()),
                "reference_fasta": self.reference_fasta_path,
                "msa_zarr_path": self.msa_zarr_path,
                "target_col": self.target_col,
                "max_context": self.max_context,
            }
        )
        if "tss_distance_group" in self.df.columns:
            base["tss_distance_group_distribution"] = self.df["tss_distance_group"].value_counts().to_dict()
            base["tss_distance_thresholds"] = self.distance_thresholds
        return base
