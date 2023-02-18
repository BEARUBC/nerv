from abc import ABC, abstractmethod
from enum import IntEnum

from src.nerv.data.dataset import EEGDatapoint


class EEGAction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class EEGController(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def analyze(self, datapoint: EEGDatapoint) -> EEGAction:
        raise NotImplementedError()
