
from email.generator import Generator
from typing import List
from data.dataset import EEGDataset, EEGDelta


def get_deltas(dtst:EEGDataset) -> Generator[EEGDelta, None, None]: 
    length = len(EEGDataset.datapoints)
    return ( EEGDataset.datapoints[i] - EEGDataset.datapoints[i-1]          for i in range(1, length)    ) 
    
