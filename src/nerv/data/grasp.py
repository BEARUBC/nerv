from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Set, Optional, Generator

import pandas as pd
import torch

from nerv.data.dataset import DatasetType, EEGDataset, EEGDatapoint
from nerv.data.loader import EEGDataLoader


@dataclass
class GraspAndLiftDatapoint(EEGDatapoint):
    """
    Datapoint selecting specific values
    """
    idx: int
    SENSOR_IDs: ClassVar[List[str]] = ["Fp1", "Fp2"]
    LABEL_IDs: ClassVar[List[str]] = ["HandStart", "FirstDigitTouch", "BothStartLoadPhase", "LiftOff", "Replace",
                                       "BothReleased"]

    @classmethod
    def from_row(cls, row: pd.Series):
        i = row["idx"]
        d = torch.tensor([row[x] for x in GraspAndLiftDatapoint.SENSOR_IDs])
        label = torch.tensor([row[x] for x in GraspAndLiftDatapoint.LABEL_IDs])
        return cls(resp=d, label=label, idx=i)


@dataclass
class GraspAndLiftDataset(EEGDataset):
    subject: int
    series: int

    # 500Hz according to https://www.kaggle.com/c/grasp-and-lift-eeg-detection/data
    sampling_rate: ClassVar[float] = 1 / 500

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, subject: int = None, series: int = None):
        datapoints = [GraspAndLiftDatapoint.from_row(df.iloc[i]) for i in range(len(df.index))]
        return cls(datapoints, subject, series)


class GraspAndLiftLoader(EEGDataLoader):
    def __init__(self, subjects: Set[int] = None, series: Set[int] = None):
        self.subjects: Optional[Set[int]] = subjects
        self.series: Optional[Set[int]] = series

    def load(self, path: Path, **kwargs) -> Generator[GraspAndLiftDataset, None, None]:
        """
        :param path:
        :param kwargs: Arguments specific to this data loader:
            - set: DatasetType, used to specify which dataset to load from the grasp and lift eeg dataset.
                There is no validation set in this dataset, so VALID will produce an error.

            - n_rows: (Optional) Number of rows to sample from each dataset
        """
        kwargs.setdefault("set", DatasetType.TRAIN)

        subdir = path

        if kwargs["set"] == DatasetType.TRAIN:
            subdir /= "train"
        elif kwargs["set"] == DatasetType.TEST:
            subdir /= "test"
        else:
            raise Exception("Validation set does not exist for the grasp and lift dataset")

        assert subdir.exists(), "Subdirectory {0} does not exist".format(subdir.resolve())

        for x in subdir.iterdir():
            name_comps = x.stem.split("_")

            # Skip all events files, we will access them in pairs from the data files
            if name_comps[-1] == "events":
                continue
            assert name_comps[-1] == "data"

            subj = int(name_comps[0][4:])
            if self.subjects and subj not in self.subjects:
                continue

            series = int(name_comps[1][6:])
            if self.series and series not in self.series:
                continue

            data_df = pd.read_csv(str(x), nrows=kwargs.get("n_rows"))
            # Read corresponding events csv
            events_df_path = subdir / ("_".join(name_comps[:-1]) + "_events.csv")
            events_df = pd.read_csv(str(events_df_path), nrows=kwargs.get("n_rows"))

            # Join the events & data
            data_df = data_df.merge(events_df, on="id")
            data_df["idx"] = data_df["id"].map(lambda a: a.split("_")[-1])
            data_df.drop(columns=["id"], inplace=True)

            yield GraspAndLiftDataset.from_dataframe(data_df, subj, series)
