from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import torch

from nerv.data.dataset import EEGBatch
from nerv.data.grasp import GraspAndLiftDatapoint


@dataclass
class EEGFrequencies:
    frequencies: Any

    # Axis 0 is the sensor, axis 1 is the frequency response at the corresponding frequency in self.frequencies
    responses: torch.tensor

    def plot(self, out_path: Path):
        l_ids = GraspAndLiftDatapoint.SENSOR_IDs
        df_data = []
        for i in range(self.responses.shape[0]):
            # Reshape existing data into a flat format for plotly
            df_data += [(self.frequencies[idx], x, l_ids[i]) for idx, x in enumerate(self.responses[i, :])]

        df = pd.DataFrame(df_data, columns=["frequency", "response", "sensor"])
        fig = px.line(df, x="frequency", y="response", color="sensor")
        fig.write_html(str(out_path))
        return fig


def eeg_batch_frequencies(batch: EEGBatch, sampling_rate=1) -> EEGFrequencies:
    n_samples = batch.resps.shape[0]
    n_sensors = batch.resps.shape[1]
    freqs = torch.fft.fftfreq(n_samples, d=sampling_rate)

    frequency_responses = torch.zeros((n_sensors, len(freqs)))
    for sensor_idx in range(n_sensors):
        data = batch.resps[:, sensor_idx]
        frequency_responses[sensor_idx, :] = torch.fft.fft(data)

    return EEGFrequencies(freqs, frequency_responses)
