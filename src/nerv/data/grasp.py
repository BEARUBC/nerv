from dataclasses import dataclass
from pathlib import Path
import random
from typing import ClassVar, List, Set

import pandas as pd
import torch

from nerv.data.dataset import DatasetType, EEGDataset, EEGDatapoint
from nerv.data.loader import EEGDataLoader


@dataclass
class GraspAndLiftDatapoint(EEGDatapoint):
    """
    Datapoint selecting specific values
    """

    subject: int
    series: int
    idx: int
    _sensor_ids: ClassVar[List[str]] = ["Fp1", "Fp2"]
    _label_ids: ClassVar[List[str]] = ["HandStart", "FirstDigitTouch", "BothStartLoadPhase", "LiftOff", "Replace",
                                       "BothReleased"]

    @classmethod
    def from_row(cls, row: pd.Series):
        subj = row["subject"]
        ser = row["series"]
        i = row["idx"]
        d = torch.tensor([row[x] for x in GraspAndLiftDatapoint._sensor_ids])
        label = torch.tensor([row[x] for x in GraspAndLiftDatapoint._label_ids])
        return cls(data=d, label=label, subject=subj, series=ser, idx=i)


@dataclass
class GraspAndLiftDataset(EEGDataset):
    subjects: Set[int]
    series: Set[int]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, subjects: Set[int] = None, series: Set[int] = None):
        datapoints = [GraspAndLiftDatapoint.from_row(df.iloc[i]) for i in range(len(df.index))]
        return cls(datapoints, subjects, series)


class GraspAndLiftLoader(EEGDataLoader):
    def load(self, path: Path, **kwargs) -> GraspAndLiftDataset:
        """

        :param path:
        :param kwargs: Arguments specific to this data loader:
            - set: DatasetType, used to specify which dataset to load from the grasp and lift eeg dataset.
                There is no validation set in this dataset, so VALID will produce an error.

            - n_rows: Number of rows to sample from each dataset

            - subjects: Set[int], list of subjects to load
            - series: Set[int], list of series to load for each loaded subject
        """
        kwargs.setdefault("set", DatasetType.TRAIN)
        kwargs.setdefault("n_rows", 10000)

        subdir = path

        if kwargs["set"] == DatasetType.TRAIN:
            subdir /= "train"
        elif kwargs["set"] == DatasetType.TEST:
            subdir /= "test"
        else:
            raise Exception("Validation set does not exist for the grasp and lift dataset")

        assert subdir.exists(), "Subdirectory {0} does not exist".format(subdir.resolve())

        out_dfs = []
        for x in subdir.iterdir():
            name_comps = x.stem.split("_")

            # Skip all events files, we will access them in pairs from the data files
            if name_comps[-1] == "events":
                continue
            assert name_comps[-1] == "data"

            subj = int(name_comps[0][4:])
            if kwargs.get("subjects"):
                if subj not in kwargs.get("subjects"):
                    continue

            series = int(name_comps[1][6:])
            if kwargs.get("series"):
                if series not in kwargs.get("series"):
                    continue

            data_df = pd.read_csv(str(x), nrows=kwargs.get("n_rows"))
            # Read corresponding events csv
            events_df_path = subdir / ("_".join(name_comps[:-1]) + "_events.csv")
            events_df = pd.read_csv(str(events_df_path), nrows=kwargs.get("n_rows"))

            # Join the events & data
            data_df = data_df.merge(events_df, on="id")
            data_df["idx"] = data_df["id"].map(lambda a: a.split("_")[-1])
            data_df.drop(columns=["id"], inplace=True)
            data_df["subject"] = subj
            data_df["series"] = series

            out_dfs.append(data_df)
        return GraspAndLiftDataset.from_dataframe(pd.concat(out_dfs), kwargs.get("subjects"), kwargs.get("series"))
