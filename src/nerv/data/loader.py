from abc import abstractmethod, ABC

from src.nerv.data.dataset import EEGDataset


class EEGDataLoader(ABC):
    @abstractmethod
    def load(self, **kwargs) -> EEGDataset:
        pass