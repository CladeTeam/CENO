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

"""Nemotron-H HF implementation (MSA-enabled, CENO-P).

Submodules are imported lazily on first access to avoid pulling in heavy
CUDA deps (mamba_ssm, flash_attn) at package-import time.
"""

from importlib import import_module

_LAZY = {
    "CENOConfig": ("configuration_ceno", "CENOConfig"),
    "CENOForCausalLM": ("modeling_ceno", "CENOForCausalLM"),
    "CENOCharLevelTokenizer": ("ceno_tokenizer", "CENOCharLevelTokenizer"),
}

__all__ = list(_LAZY.keys())


def __getattr__(name):
    if name in _LAZY:
        mod_path, cls_name = _LAZY[name]
        mod = import_module(f".{mod_path}", __name__)
        obj = getattr(mod, cls_name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
