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

"""CENO-P model adapter for VEP scoring.

Wraps the MSA-enabled Nemotron-H causal LM (``modeling_ceno_p``)
behind the :class:`~vep.base_model.BaseModel` interface that the
:class:`~vep.evaluator.VEPEvaluator` expects.

Input contract (matches :class:`~vep.dataset.CENOMSADataset`):
  each "sequence" is a *packed* bytes blob encoding
  ``(token_ids: List[int], segment_sizes: List[int])`` — the MSA rows
  concatenated into one byte-level token stream, with the target (reference)
  sequence as the last segment. The adapter unpacks this, builds a per-token
  ``seq_idx`` (segment id), and scores only the **last segment**'s
  next-token log-likelihood.

Plain ``str`` inputs are also accepted (single segment, no MSA) for debugging.
"""
import os
import struct
import sys
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from .base_model import BaseModel

Pooling = Literal["mean", "max", "cls"]
TargetSegment = Literal["all", "last"]

MSAInput = Tuple[List[int], List[int]]
PackedInput = Union[bytes, bytearray, memoryview]


class CENOMSAModel(BaseModel):
    """CENO-P (MSA) scorer.

    Args:
        model_name: tag for result logging (e.g. ``"CENO-P-1B"``).
        model_path: checkpoint directory (config.json + weights).
        pad_id: pad token id (byte-level tokenizer -> 1).
        torch_dtype: load dtype; ``None`` -> read ``torch_dtype`` from config.
        device_map: HF device_map; ``"auto"`` uses accelerate.
        target_segment: ``"last"`` scores only the target segment of each
            packed MSA input (the VEP scoring convention); ``"all`` scores
            every token.
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        pad_id: int = 1,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, dict]] = "auto",
        target_segment: TargetSegment = "last",
    ):
        super().__init__(model_name, model_path)
        self.pad_id = int(pad_id)
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.target_segment: TargetSegment = target_segment
        self._load_model()

    def _load_model(self):
        # Make the ceno_model package importable regardless of CWD.
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from ceno_model.ceno_hf.configuration_ceno import CENOConfig
        from ceno_model.ceno_hf.modeling_ceno_p import (
            CENOPForCausalLM,
        )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"model_path not found: {self.model_path}")

        cfg = CENOConfig.from_pretrained(self.model_path)
        dtype = self.torch_dtype
        if dtype is None:
            cfg_dtype = getattr(cfg, "torch_dtype", None)
            if isinstance(cfg_dtype, str):
                dtype = getattr(torch, cfg_dtype, None)
            elif isinstance(cfg_dtype, torch.dtype):
                dtype = cfg_dtype
        if dtype is None:
            dtype = torch.float32

        load_kwargs = {
            "config": cfg,
            "torch_dtype": dtype,
            "device_map": self.device_map if torch.cuda.is_available() else None,
        }
        # Optional FlashAttention-2 (env-driven; safe to leave unset).
        attn_impl = os.environ.get("CENO_ATTN_IMPLEMENTATION", "").strip()
        if not attn_impl:
            enable_fa2 = os.environ.get("CENO_ENABLE_FLASH_ATTENTION_2", "").strip().lower()
            if enable_fa2 in ("1", "true", "yes"):
                attn_impl = "flash_attention_2"
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl
            print(f"[INFO] CENO attn_implementation = {attn_impl}", flush=True)

        self.model = CENOPForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        self.model.eval()

        # Optional Mamba CUDA kernels (env-driven; PyTorch fallback otherwise).
        enable_mamba_kernels = os.environ.get(
            "CENO_ENABLE_MAMBA_KERNELS", ""
        ).strip().lower() in ("1", "true", "yes")
        if enable_mamba_kernels:
            self.model.config.use_mamba_kernels = True
            print("[INFO] CENO use_mamba_kernels = True", flush=True)

        try:
            self._device = next(self.model.parameters()).device
        except StopIteration:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ #
    # Packing / unpacking (must match vep.dataset._pack_token_ids_and_segments)
    # ------------------------------------------------------------------ #
    def _encode_ascii_bytes(self, seq: str) -> List[int]:
        if not seq:
            return []
        b = np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)
        return b.astype(np.int64).tolist()

    def _unpack_packed(self, packed: PackedInput) -> MSAInput:
        """Unpack the bytes blob produced by the dataset.

        Format (little-endian):
          uint16  n_segments
          uint32  seg_0_len ... seg_{n-1}_len
          uint32  n_token_bytes
          bytes   token_bytes
        """
        buf = bytes(packed)
        if len(buf) < 2 + 4:
            return ([], [])
        off = 0
        (nseg,) = struct.unpack_from("<H", buf, off)
        off += 2
        segs: List[int] = []
        if nseg:
            segs = list(struct.unpack_from(f"<{nseg}I", buf, off))
            off += 4 * nseg
        (nbytes,) = struct.unpack_from("<I", buf, off)
        off += 4
        token_bytes = buf[off : off + nbytes]
        token_ids = list(np.frombuffer(token_bytes, dtype=np.uint8).astype(np.int64))
        return token_ids, [int(x) for x in segs]

    def _build_seq_idx(self, segment_sizes: List[int]) -> List[int]:
        seq_idx: List[int] = []
        cur_id = 0
        for seg_len in segment_sizes:
            if seg_len <= 0:
                continue
            seq_idx.extend([cur_id] * int(seg_len))
            cur_id += 1
        return seq_idx

    def _prepare_batch(
        self, sequences: List[Union[str, MSAInput, PackedInput]]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[List[int]]]:
        """Returns input_ids (B,L), attention_mask (B,L), seq_idx (B,L)|None, segs_list."""
        ids_list: List[List[int]] = []
        segs_list: List[List[int]] = []

        for item in sequences:
            if isinstance(item, (bytes, bytearray, memoryview)):
                ids, segs = self._unpack_packed(item)
            elif isinstance(item, str):
                ids = self._encode_ascii_bytes(item)
                segs = [len(ids)]
            else:
                ids, segs = item
                ids = list(ids)
                segs = list(segs)
            ids_list.append(ids)
            segs_list.append(segs)

        max_len = max((len(x) for x in ids_list), default=0)
        if max_len <= 0:
            empty = torch.empty((len(ids_list), 0), dtype=torch.long, device=self._device)
            return empty, empty, None, segs_list

        input_ids = torch.full((len(ids_list), max_len), self.pad_id, dtype=torch.long, device=self._device)
        attention_mask = torch.zeros((len(ids_list), max_len), dtype=torch.long, device=self._device)
        seq_idx = torch.zeros((len(ids_list), max_len), dtype=torch.int32, device=self._device)

        for b, (ids, segs) in enumerate(zip(ids_list, segs_list)):
            n = len(ids)
            if n == 0:
                continue
            input_ids[b, :n] = torch.tensor(ids, dtype=torch.long, device=self._device)
            attention_mask[b, :n] = 1
            seq_idx_row = self._build_seq_idx(segs)
            pad_seq_id = (max(seq_idx_row) + 1) if seq_idx_row else 0
            seq_idx_row = seq_idx_row[:n] + [pad_seq_id] * max(0, max_len - n)
            seq_idx[b] = torch.tensor(seq_idx_row, dtype=torch.int32, device=self._device)

        return input_ids, attention_mask, seq_idx, segs_list

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def score_sequences(self, sequences: List, batch_size: int = 256) -> List[float]:
        all_scores: List[float] = []
        if not sequences:
            return all_scores

        with torch.no_grad():
            for st in tqdm(range(0, len(sequences), batch_size), desc="Scoring (CENO-P)"):
                batch = sequences[st : st + batch_size]
                all_scores.extend(self._score_batch(batch))
        return all_scores

    def _score_batch(self, sequences: List) -> List[float]:
        input_ids, attention_mask, seq_idx, segs_list = self._prepare_batch(sequences)
        if input_ids.numel() == 0:
            return [0.0 for _ in sequences]

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            seq_idx=seq_idx,
            use_cache=False,
        )
        logits = out.logits  # (B, L, V)
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # (B, L-1, V)
        target_ids = input_ids[:, 1:]  # (B, L-1)
        tok_lp = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
        valid_mask = attention_mask[:, 1:].to(tok_lp.dtype)  # (B, L-1)

        scores: List[float] = []
        for b in range(tok_lp.shape[0]):
            lp_row = tok_lp[b]
            mask_row = valid_mask[b]

            if self.target_segment == "last":
                segs = [int(x) for x in segs_list[b] if int(x) > 0]
                if len(segs) > 1:
                    start = int(sum(segs[:-1]))
                    end = int(sum(segs))
                    # target positions are 1..L-1; prediction index i predicts target at i+1
                    pred_start = max(0, start - 1)
                    pred_end = max(0, end - 1)
                    seg_mask = torch.zeros_like(mask_row)
                    seg_mask[pred_start:pred_end] = 1
                    mask_row = mask_row * seg_mask

            denom = float(mask_row.sum().item())
            if denom <= 0:
                scores.append(0.0)
            else:
                scores.append(float((lp_row * mask_row).sum().item() / denom))
        return scores

    @torch.no_grad()
    def get_embedding(
        self,
        sequences: List,
        layer_name: str,
        batch_size: int = 64,
        pool: Pooling = "mean",
        return_numpy: bool = True,
    ):
        """Minimal embedding API (used for hidden-size inference only)."""
        if isinstance(sequences, (str, tuple, bytes, bytearray, memoryview)):
            sequences = [sequences]

        embs = []
        with torch.no_grad():
            for st in tqdm(range(0, len(sequences), batch_size), desc="Embedding (CENO-P)"):
                batch = sequences[st : st + batch_size]
                input_ids, attention_mask, seq_idx, _ = self._prepare_batch(batch)
                if input_ids.numel() == 0:
                    continue
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    seq_idx=seq_idx,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = out.hidden_states
                if not hidden_states:
                    raise RuntimeError("Model did not return hidden_states.")
                hs = hidden_states[-1]
                if layer_name and layer_name != "hidden_states":
                    n = layer_name.strip().lower()
                    if n.startswith("layer_"):
                        try:
                            hs = hidden_states[int(n.split("_", 1)[1])]
                        except Exception:
                            hs = hidden_states[-1]
                attn = attention_mask.to(torch.bool)
                if pool == "cls":
                    pooled = hs[:, 0, :]
                elif pool == "max":
                    masked = hs.masked_fill(~attn.unsqueeze(-1), float("-inf"))
                    pooled, _ = masked.max(dim=1)
                    bad = ~torch.isfinite(pooled).all(dim=1)
                    if bad.any():
                        pooled[bad] = 0
                else:
                    mask_f = attn.unsqueeze(-1).to(hs.dtype)
                    pooled = (hs * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
                embs.append(pooled.float().cpu().numpy() if return_numpy else pooled.float().cpu())

        if not embs:
            return np.zeros((0, 0), dtype=np.float32) if return_numpy else torch.zeros((0, 0))
        return np.concatenate(embs, axis=0) if return_numpy else torch.cat(embs, dim=0)
