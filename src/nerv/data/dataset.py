from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import torch


class DatasetType(Enum):
    UNDEFINED = auto()
    TRAIN = auto()
    VALID = auto()
    TEST = auto()


@dataclass
class EEGDatapoint(ABC):
    data: torch.tensor
    label: Optional[torch.tensor]


@dataclass
class EEGDataset:
    datapoints: List[EEGDatapoint]

    def batch_tensor(self, start_idx: int, end_idx: int) -> torch.tensor:
        return torch.cat([x.data for x in self.datapoints[start_idx:end_idx]])
