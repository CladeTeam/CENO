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

"""Smoke test for the ceno_model package — needs NO checkpoint weights.

Verifies that:
  1. The config class loads and exposes the CENO-specific fields
     (``intra_encoding_pattern``/``intra_encoding_mask``, ``hybrid_override_pattern``).
  2. The byte-level tokenizer encodes DNA to ASCII byte ids.
  3. Both modeling modules (base + MSA training path) import cleanly.

Run from the repo root:
    python -m ceno_model.examples.load_model
"""
import os
import sys

# Make ``ceno_model`` importable when run as a script / module from repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ceno_model.ceno_hf import CENOConfig, CENOCharLevelTokenizer
from ceno_model.ceno_hf import modeling_ceno  # noqa: F401  (import side-effect)
from ceno_model.ceno_hf import modeling_ceno_p  # noqa: F401


def main():
    print("=" * 64)
    print("ceno_model smoke test (no weights required)")
    print("=" * 64)

    # --- 1. Config ---------------------------------------------------------
    # Instantiate with CENO-P-ish defaults (300M-class). A real checkpoint's
    # config.json overrides all of these via from_pretrained.
    cfg = CENOConfig(
        vocab_size=512,
        hidden_size=1024,
        num_hidden_layers=9,
        hybrid_override_pattern="MEM*EMEME",
        intra_encoding_pattern="++---++--",
    )
    print("\n[Config] CENOConfig (CENO-P-like 9-layer example)")
    print(f"  model_type            = {cfg.model_type}")
    print(f"  vocab_size            = {cfg.vocab_size}")
    print(f"  hidden_size           = {cfg.hidden_size}")
    print(f"  num_hidden_layers     = {cfg.num_hidden_layers}")
    print(f"  hybrid_override_pattern = {cfg.hybrid_override_pattern!r}")
    print(f"  intra_encoding_pattern  = {cfg.intra_encoding_pattern!r}")
    mask = cfg.intra_encoding_mask
    print(f"  intra_encoding_mask     = {mask}  (+ = isolate MSA rows, - = fuse)")

    assert len(cfg.hybrid_override_pattern) == cfg.num_hidden_layers
    assert len(mask) == cfg.num_hidden_layers
    assert sum(mask) == 4  # four '+' in '++---++--'

    # --- 2. Tokenizer ------------------------------------------------------
    tok = CENOCharLevelTokenizer(vocab_size=512)
    seq = "ATCGATCG"
    ids = tok.encode(seq, add_special_tokens=False)
    expected = [ord(c) for c in seq]  # byte-level: A=65 T=84 C=67 G=71
    print("\n[Tokenizer] CENOCharLevelTokenizer (byte-level, vocab=512)")
    print(f"  seq      = {seq!r}")
    print(f"  token_ids = {ids}")
    print(f"  expected  = {expected}  (ASCII bytes)")
    assert ids == expected, f"tokenizer mismatch: {ids} != {expected}"
    decoded = tok.decode(ids)
    print(f"  decoded   = {decoded!r}")
    assert decoded == seq

    # --- 3. Modeling modules import ---------------------------------------
    print("\n[Modeling] imports OK:")
    print(f"  modeling_ceno.CENOForCausalLM            = "
          f"{modeling_ceno.CENOForCausalLM.__name__}")
    print(f"  modeling_ceno_p.CENOPForCausalLM         = "
          f"{modeling_ceno_p.CENOPForCausalLM.__name__}")

    print("\n" + "=" * 64)
    print("PASS — ceno_model package is importable & functional.")
    print("Load real weights with: CENOForCausalLM.from_pretrained('<ckpt_dir>')")
    print("=" * 64)


if __name__ == "__main__":
    main()
