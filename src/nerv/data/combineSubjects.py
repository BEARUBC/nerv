import os
from pathlib import Path
from src.nerv.definitions import DATA_PATH

import numpy as np
import pandas as pd
import mne.channels
from mne import create_info


class EEGDataset:
    def __init__(self, epoch_array, sampling_freq):
        self.epoch_array = epoch_array
        self.sampling_freq = sampling_freq


def _load(path: Path):
    data = pd.read_csv(os.path.abspath(path), header=None)

    col_names = pd.read_csv(DATA_PATH / "columnLabels.csv")
    ch_names = list(col_names.columns[4:])
    ch_type = ['eeg'] * 64 + ['eog'] * 4 + ['misc'] + ['eeg']
    info = create_info(ch_names, 1024, ch_type)

    npdata = np.array(data)
    # make data of all subjects the same size so that the same number of events is found
    # slice data
    npdata = npdata[:844800, :] # min number of rows in any subject
    onsets = np.array(np.where(npdata[:, 3] == 1537))
    conditions = npdata[npdata[:, 3] == 1537, 2]
    events = np.squeeze(np.dstack((onsets.flatten(), np.zeros(conditions.shape), conditions))).astype('int')

    return npdata, info, events


class CombineSubjects:
    def __init__(self):
        self.tmin = -1.5
        self.sfreq = 1024
        self.event_id = dict(Button_tone=1, Tone_only=2, Button_only=3)

    def load(self, path: Path) -> EEGDataset:
        if path.is_file():
            data, info, events = _load(path)
            dtst = mne.EpochsArray(data, info, events, self.tmin, self.event_id)
            return EEGDataset(dtst, sampling_freq=self.sfreq)
        elif path.is_dir():
            files = path.glob("**/*")
            data, info, events = None, None, None
            n_files = 0

            for idx, file in enumerate(files):
                print(file)
                if not file.is_file():
                    continue
                print("loading file ", file)
                _data, _info, _events = _load(file)
                n_files += 1
                if n_files == 1:
                    info = _info
                    events = _events
                    data = _data
                    continue

                data += _data
                print(data)
            data /= n_files
            EEGdata = data.reshape(len(events), 3072, 74)
            # remove the first 4 columns (non-eeg):
            EEGdata = EEGdata[:, :, 4:]
            EEGdata = np.swapaxes(EEGdata, 1, 2)
            dtst = mne.EpochsArray(EEGdata, info, events, self.tmin, self.event_id)
            return EEGDataset(dtst, sampling_freq=self.sfreq)

# subjects = CombineSubjects()
# subjects_combined = CombineSubjects.load(subjects, Path('C:/Users/kiray/PycharmProjects/mne/data/'))
# get separate epochs arrays for controls and patients
# demographics = pd.read_csv('C:/Users/kiray/PycharmProjects/mne/demographic.csv')
# controls = np.array(demographics.loc[demographics['group'] == 0, 'subject'])
# patients = np.array(demographics.loc[demographics['group'] == 1, 'subject'])


controls = CombineSubjects()
controls_combined = CombineSubjects.load(controls, Path(DATA_PATH / "controls"))
patients = CombineSubjects()
patients_combined = CombineSubjects.load(patients, Path(DATA_PATH / "patients"))

controls_evoked = controls_combined.epoch_array.average(by_event_type=True)
patients_evoked = patients_combined.epoch_array.average(by_event_type=True)

patient_evoked_button_tone = patients_evoked[0]
patient_evoked_tone = patients_evoked[1]
patient_evoked_button = patients_evoked[2]
control_evoked_button_tone = controls_evoked[0]
control_evoked_tone = controls_evoked[1]
control_evoked_button = controls_evoked[2]

# mne evoked built-in visualization, combines all sensors in various ways
button_tone_evokes = [patient_evoked_button_tone, control_evoked_button_tone]
for combine in ('mean', 'median', 'gfp', 'std'):
    mne.viz.plot_compare_evokeds(button_tone_evokes, picks='eeg', combine=combine)

tone_evokes = [patient_evoked_tone, control_evoked_tone]
for combine in ('mean', 'median', 'gfp', 'std'):
    mne.viz.plot_compare_evokeds(tone_evokes, picks='eeg', combine=combine)






