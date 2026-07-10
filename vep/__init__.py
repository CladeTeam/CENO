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

"""CENO Variant Effect Prediction (VEP) — HuggingFace-only, TraitGym example.

Pipeline: CENO-P model adapter + MSA dataset (zarr) + delta-log-likelihood
evaluator (AUROC / AUPRC). No vLLM scoring — pure HuggingFace forward.
"""

from .base_model import BaseModel
from .base_dataset import BaseDataset, TaskType
from .model_adapter import CENOMSAModel
from .dataset import CENOMSADataset
from .evaluator import VEPEvaluator

__all__ = [
    "BaseModel",
    "BaseDataset",
    "TaskType",
    "CENOMSAModel",
    "CENOMSADataset",
    "VEPEvaluator",
]
