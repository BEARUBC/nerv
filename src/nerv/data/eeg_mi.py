import numpy as np
import mne
import glob
from scipy.io import loadmat

from src.nerv.definitions import DATA_PATH

dataset_path = DATA_PATH/"sess01_subj01_EEG_MI.mat"
dataset = loadmat(str(dataset_path))

# x = continous EEG signals (data points/samples * channels)
# t = stimulus onset times of each trial
# fs = sampling rates (aka sampling frequency)
# y_dec = class labels in integer types
# y_logic = class labels in logical types
# y_class = class definitions
# chan = channel information
# smt = time * trials * channels

# We need the following information to create MNE structure:
#   data ([ndarray]): [trials x chans x samples]
#   y ([ndarray]):    [class label array  [1, labels]]
#   sfreq ([int]):    [sampling frequency]
#   event_id ([dict]): [{1 :'pos', -1 : 'neg'} - class labels id]
#   chan_names ([list]): [channel names in a list of strings]

RAW_chan_names = dataset['EEG_MI_test']['chan'][0][0][0]
chan_names = [str(x[0]) for x in RAW_chan_names]

event_id = dict(Left_Hand=2, Right_Hand=1)
sfreq = 1000

n_channels = len(chan_names)

# Initialize an info structure
info = mne.create_info(ch_names=chan_names, sfreq=sfreq, ch_types='eeg')

# Prints the list of all standard montages shipping with MNE-Python
print(mne.channels.get_builtin_montages(descriptions=True))

# set_montage() function takes in a montage. 'standard_1020' must be one of the built-in montages; using a
# string as input will update the channel information with the channel positions in the montage
# Read more about montages here: https://mne.tools/dev/auto_tutorials/intro/40_sensor_locations.html
# Used 'standard_1005' instead of 'standard_1020' because it did not have all the channels in the MATLAB dataset
info.set_montage('standard_1005')

# Y is the first row of y_dec
Y = dataset['EEG_MI_test']['y_dec'][0][0][0]
labels = np.array(Y)
ev_tests = [i * 4000 for i in range(Y.shape[0])]
ev = np.array(ev_tests)
eventLength = 100
events = np.column_stack((ev,
                          np.zeros(eventLength, dtype=int),
                          labels))

X = dataset['EEG_MI_test']['smt'][0][0]
data = np.moveaxis(X, 0, 2)
tmin = 0
epochs = mne.EpochsArray(data, info, events, tmin, event_id)

# Analysis
epochs.average().plot()
epochs.plot_psd()