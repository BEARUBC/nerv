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

        sfreq = int(dataset['o']['sampFreq'][0][()][0][0])
        raw_chan_names = dataset['o']['chnames'][0][0][()][:-1]  # Excess channel is cut out
        chan_names = np.array([x[0][0] for x in raw_chan_names])

        info = mne.create_info(ch_names=chan_names.tolist(), sfreq=sfreq, ch_types='eeg')
        info.set_montage('standard_1020')

        data = np.array(dataset['o']['data'][()][0][0][:, :-1]).transpose()  # May need to cut this to onset time and
        # 1500 ms after
        raw = mne.io.RawArray(data, info)
        stim = dataset['o']['marker'][0][()][0]
        events = mne.find_events(data, stim)

        # Y = dataset['SMT']['y_dec'][0][0][0]
        # labels = np.array(Y)
        # ev_tests = [i * 4000 for i in range(Y.shape[0])]
        # ev = np.array(ev_tests)
        # eventLength = 100
        # events = np.column_stack((ev,
        #                           np.zeros(eventLength, dtype=int),
        #                           labels))

        # X = dataset['SMT']['x'][0][0]
        # RAW_chan_names = dataset['SMT']['chan'][0][0][0]
        # chan_names = np.array([str(x[0]) for x in RAW_chan_names])
        # montage1020 = mne.channels.make_standard_montage('standard_1020')
        # keep_mask = [True if x in montage1020.ch_names else False for x in chan_names]
        # chan_names = chan_names[keep_mask]
        # X = X[:, :, keep_mask]

        # pre_rest = dataset['EEG_MI_test']['pre_rest'][0][0]
        # pre_rest = dataset['SMT']['pre_rest'][0][0]
        # pre_rest = pre_rest[:, keep_mask]
        # rest_means = np.reshape(np.mean(pre_rest[-1000:], axis=0), (1, 1, -1))
        # X -= rest_means
        # # Microvolts to volts for MNE
        # X /= 10 ** 6
        # data = np.moveaxis(X, 0, 2)

        return raw, events, sfreq

    def load(self, path: Path) -> EEGDataset:
        if path.is_file():
            raw, events, sfreq = self._load(path)
            # dtst = mne.EpochsArray(data, info, events, self.event_id)
            dtst = mne.EpochsArray(raw, events, event_id=self.event_id, preload=True)
            # not sure about start and end times or reject dict
            return EEGDataset(dtst, sampling_freq=self.sfreq)
        elif path.is_dir():
            files = path.glob("**/*")
            raw, events = None, None
            n_files = 0

            for idx, file in enumerate(files):
                print(file)
                if not file.is_file():
                    continue
                print("loading file ", file)
                _raw, _events, sfreq = self._load(file)
                n_files += 1
                if n_files == 1:
                    raw = _raw
                    events = _events
                    continue

                raw += _raw
                print(raw)
            raw /= n_files
            dtst = mne.EpochsArray(raw, events, event_id=self.event_id, preload=True)
            # dtst = mne.EpochsArray(data, info, events, self.tmin, self.event_id)
            return EEGDataset(dtst, sfreq)  # sfreq is subject to change, not sure if this pipeline will work

    def __init__(self):
        self.event_id = dict(Left_Hand=1, Right_Hand=2, Neutral=3)
