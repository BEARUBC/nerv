import numpy as np

from src.nerv.control.controller import EEGController, EEGAction
from src.nerv.data.dataset import EEGDatapoint


class StandardController(EEGController):
    def analyze(self, datapoint: EEGDatapoint) -> EEGAction:
        pass

