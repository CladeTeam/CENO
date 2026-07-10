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

"""Dataset base class for VEP."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

from torch.utils.data import Dataset


class TaskType(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class BaseDataset(Dataset, ABC):
    """Base dataset supporting regression / classification tasks."""

    def __init__(self, data_path: str, task_type: TaskType, window_size: Optional[int] = None):
        self.data_path = data_path
        self.task_type = task_type
        self.window_size = window_size
        self.data = None

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple:
        """REGRESSION -> (wt, mt, fitness: float); CLASSIFICATION -> (wt, mt, label: int)."""
        raise NotImplementedError

    def get_task_type(self) -> TaskType:
        return self.task_type

    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    def is_classification(self) -> bool:
        return self.task_type == TaskType.CLASSIFICATION

    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "data_path": self.data_path,
            "task_type": self.task_type.value,
            "window_size": self.window_size,
            "dataset_size": len(self),
            "dataset_class": self.__class__.__name__,
        }
