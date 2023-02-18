from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat

from src.nerv.data.dataset import EEGDataset
from src.nerv.data.loader import EEGDataLoader


class MatlabLoader(EEGDataLoader):
    def _load(self, path: Path):
        dataset = loadmat(str(path))

        # x = continous EEG signals (data points/samples * channels)
        # t = stimulus onset times of each trial
        # fs = sampling rates (aka sampling frequency)
        # y_dec = class labels in integer types
        # y_logic = class labels in logical types
        # y_class = class definitions
        # chan = channel information
        # smt = time * trials * channels
        # pre_rest = time * channels

        # We need the following information to create MNE structure:
        #   data ([ndarray]): [trials x chans x samples]
        #   y ([ndarray]):    [class label array  [1, labels]]
        #   sfreq ([int]):    [sampling frequency]
        #   event_id ([dict]): [{1 :'pos', -1 : 'neg'} - class labels id]
        #   chan_names ([list]): [channel names in a list of strings]

        # RAW_chan_names = dataset['avSMT']['chan'][0][0][0]
        # chan_names = np.array([str(x[0]) for x in RAW_chan_names])
        # montage1020 = mne.channels.make_standard_montage('standard_1020')
        # keep_mask = [True if x in montage1020.ch_names else False for x in chan_names]
        # chan_names = chan_names[keep_mask]
        # x = dataset['avSMT']['x'][0][0]
        # x = x[:, :, keep_mask]
        # info = mne.create_info(ch_names=chan_names.tolist(), sfreq=self.sfreq, ch_types='eeg')
        # info.set_montage('standard_1020')
        # left_average = np.moveaxis(x[:, 0, :], 0, 1)
        # right_average = np.moveaxis(x[:, 1, :], 0, 1)
        # left_evoked = mne.EvokedArray(left_average, info, nave=100)
        # right_evoked = mne.EvokedArray(right_average, info, nave=100)
        # evokeds = dict(Left=left_evoked, Right=right_evoked)
        # picks1 = 'C3'
        # mne.viz.plot_compare_evokeds(evokeds, picks=picks1, colors=['r', 'b'])

        # RAW_chan_names = dataset['EEG_MI_test']['chan'][0][0][0]
        # RAW_chan_names = dataset['SMT']['chan'][0][0][0]
        # chan_names = np.array([str(x[0]) for x in RAW_chan_names])

        # Y is the first row of y_dec
        # Y = dataset['EEG_MI_test']['y_dec'][0][0][0]
        Y = dataset['SMT']['y_dec'][0][0][0]
        labels = np.array(Y)
        ev_tests = [i * 4000 for i in range(Y.shape[0])]
        ev = np.array(ev_tests)
        eventLength = 100
        events = np.column_stack((ev,
                                  np.zeros(eventLength, dtype=int),
                                  labels))

        # events_rest = np.column_stack((np.array(0),
        #                                np.array(0),
        #                                np.array(0)))

        # X = dataset['EEG_MI_test']['smt'][0][0]
        X = dataset['SMT']['x'][0][0]
        RAW_chan_names = dataset['SMT']['chan'][0][0][0]
        chan_names = np.array([str(x[0]) for x in RAW_chan_names])
        montage1020 = mne.channels.make_standard_montage('standard_1020')
        keep_mask = [True if x in montage1020.ch_names else False for x in chan_names]
        chan_names = chan_names[keep_mask]
        X = X[:, :, keep_mask]

        # pre_rest = dataset['EEG_MI_test']['pre_rest'][0][0]
        pre_rest = dataset['SMT']['pre_rest'][0][0]
        pre_rest = pre_rest[:, keep_mask]
        rest_means = np.reshape(np.mean(pre_rest[-1000:], axis=0), (1, 1, -1))
        X -= rest_means
        # Microvolts to volts for MNE
        X /= 10 ** 6
        data = np.moveaxis(X, 0, 2)

        # TODO: Impute null values here

        # Initialize an info structure
        info = mne.create_info(ch_names=chan_names.tolist(), sfreq=self.sfreq, ch_types='eeg')
        info.set_montage('standard_1020')
        return data, rest_means, info, events

    def load(self, path: Path) -> EEGDataset:
        if path.is_file():
            data, rest_means, info, events = self._load(path)
            dtst = mne.EpochsArray(data, info, events, self.tmin, self.event_id)
            return EEGDataset(dtst, sampling_freq=self.sfreq)
        elif path.is_dir():
            files = path.glob("**/*")
            data, rest_means, info, events = None, None, None, None
            n_files = 0

            for idx, file in enumerate(files):
                print(file)
                if not file.is_file():
                    continue
                print("loading file ", file)
                _data, _rest, _info, _events = self._load(file)
                n_files += 1
                if n_files == 1:
                    info = _info
                    events = _events
                    data = _data
                    rest_means = _rest
                    continue

                data += _data
                rest_means += _rest
                print(data)
            data /= n_files
            rest_means /= n_files
            dtst = mne.EpochsArray(data, info, events, self.tmin, self.event_id)
            return EEGDataset(dtst, sampling_freq=self.sfreq)

    def __init__(self):
        self.tmin = 0
        # self.sfreq = 1000
        self.sfreq = 100
        self.event_id = dict(Left_Hand=2, Right_Hand=1)
        self.pre_rest_id = dict(Pre_Rest=0)  # ?
