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

from abc import ABC, abstractmethod
from typing import List


class BaseModel(ABC):
    """Model interface every VEP scorer must implement.

    The evaluator only depends on :meth:`score_sequences`: given a list of
    packed MSA inputs (bytes) or plain sequences (str), return one float
    score per input (mean next-token log-likelihood of the target segment).
    """

    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path

    @abstractmethod
    def score_sequences(self, sequences: List, batch_size: int = 256) -> List[float]:
        """Score a batch of sequences. Output length == input length, same order."""
        raise NotImplementedError

    def get_embedding(self, sequences: List, layer_name: str, batch_size: int = 64):
        """Optional embedding API (used only for hidden-size inference)."""
        raise NotImplementedError
