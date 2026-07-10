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

"""CENO model code package (HuggingFace trust_remote_code compatible).

Exposes the Nemotron-H hybrid (Mamba/Attention/MoE) implementation with the
MSA (multi-species alignment) post-training additions used by CENO-P:

- ``configuration_ceno.CENOConfig``  — config, including
  ``intra_encoding_pattern`` (the ``+``/``-`` per-layer MSA isolation flag).
- ``modeling_ceno.CENOForCausalLM``  — base causal-LM forward,
  used for generation and single-sequence scoring (``seq_idx`` optional).
- ``CENOPForCausalLM`` — same model
  with the MSA scoring path that consumes a per-token ``seq_idx`` (packed
  MSA rows) and applies ``intra_encoding_mask`` per layer. Used by the VEP
  adapter in :mod:`vep.model_adapter`.
- ``ceno_tokenizer.CENOCharLevelTokenizer`` — byte-level tokenizer.

These files are designed to be loaded by ``transformers`` with
``trust_remote_code=True`` from a checkpoint directory (via ``auto_map``),
or imported directly as a Python package.
"""

__all__ = ["ceno_hf"]
