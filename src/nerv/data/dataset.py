from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Generator, Callable

import torch


class DatasetType(Enum):
    UNDEFINED = auto()
    TRAIN = auto()
    VALID = auto()
    TEST = auto()


@dataclass
class EEGDatapoint(ABC):
    resp: torch.tensor
    label: torch.tensor


@dataclass
class EEGBatch(ABC):
    resps: torch.tensor
    labels: torch.tensor



@dataclass
class EEGDataset:
    datapoints: List[EEGDatapoint]

    def batch(self, start_idx: int, end_idx: int) -> EEGBatch:
        """
        Creates a batch from the existing data points, in a tensor form.
        :param start_idx: Start index within self.datapoints to begin the batch
        :param end_idx: End index within self.datapoints to end the batch
        :return: The created EEGBatch object
        """
        dps = self.datapoints[start_idx:end_idx]
        resps = torch.stack([x.resp for x in dps])
        labels = torch.stack([x.label for x in dps])
        return EEGBatch(resps, labels)

    def rolling_window(self, size: int, stride: int = 1, start_idx: int = 0, end_idx: int = None) -> Generator[
        EEGBatch, None, None]:
        """
        Creates rolling window batches across the EEG data. Returns a generator of the created batches.
        :param size: Size of each rolling window batch
        :param stride: Distance between the batch start indices
        :param start_idx: Optional parameter to not create rolling windows for the entire
        :param end_idx:
        """
        if not end_idx:
            end_idx = len(self.datapoints)

        for i in range(start_idx, end_idx, stride):
            if i + size > len(self.datapoints):
                return
            yield self.batch(i, i + size)

    def select(self, criteria: Callable[[EEGDatapoint], bool]) -> "EEGDataset":
        """
        Select all datapoints that match a specified criteria.
        :param criteria: Callable that returns a boolean specifying whether a specific datapoint matches the criteria
        :return: a new EEGDataset only containing the datapoints matching the criteria
        """
        return EEGDataset([x for x in self.datapoints if criteria(x)])
