from abc import abstractmethod, ABC
from pathlib import Path


class EEGDataLoader(ABC):

    @abstractmethod
    def load(self, path: Path, **kwargs):
        pass
