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
# pre_rest = time * channels

# We need the following information to create MNE structure:
#   data ([ndarray]): [trials x chans x samples]
#   y ([ndarray]):    [class label array  [1, labels]]
#   sfreq ([int]):    [sampling frequency]
#   event_id ([dict]): [{1 :'pos', -1 : 'neg'} - class labels id]
#   chan_names ([list]): [channel names in a list of strings]

RAW_chan_names = dataset['EEG_MI_test']['chan'][0][0][0]
chan_names = np.array([str(x[0]) for x in RAW_chan_names])

event_id = dict(Left_Hand=2, Right_Hand=1)
pre_rest_id = dict(Pre_Rest=0)
sfreq = 1000

# Y is the first row of y_dec
Y = dataset['EEG_MI_test']['y_dec'][0][0][0]
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

X = dataset['EEG_MI_test']['smt'][0][0]
montage1020 = mne.channels.make_standard_montage('standard_1020')
keep_mask = [True if x in montage1020.ch_names else False for x in chan_names]
chan_names = chan_names[keep_mask]
X = X[:, :, keep_mask]
pre_rest = dataset['EEG_MI_test']['pre_rest'][0][0]
pre_rest = pre_rest[:, keep_mask]
rest_means = np.reshape(np.mean(pre_rest, axis=0), (1, 1, -1))
X -= rest_means
# Microvolts to volts for MNE
X /= 10**6
data = np.moveaxis(X, 0, 2)

# Initialize an info structure
info = mne.create_info(ch_names=chan_names.tolist(), sfreq=sfreq, ch_types='eeg')

# set_montage() function takes in a montage. 'standard_1020' must be one of the built-in montages; using a
# string as input will update the channel information with the channel positions in the montage
# Read more about montages here: https://mne.tools/dev/auto_tutorials/intro/40_sensor_locations.html
# Used 'standard_1005' instead of 'standard_1020' because it did not have all the channels in the MATLAB dataset
info.set_montage('standard_1020')

tmin = 0
epochs_action = mne.EpochsArray(data, info, events, tmin, event_id)

# MNE analysis with epochs plots
# epochs.average().plot()
# epochs.plot_psd()

# MNE analysis with evoked objects
evoked_left = epochs_action['Left_Hand'].average()
evoked_right = epochs_action['Right_Hand'].average()
evokeds = dict(Left=evoked_left, Right=evoked_right)
picks1 = 'C3'
mne.viz.plot_compare_evokeds(evokeds, picks=picks1)

picks2 = 'C4'
mne.viz.plot_compare_evokeds(evokeds, picks=picks2)
# Experiment used grand-averaged brain response, which likely is the average across all subjects
# mne.grand_average() should be used in this scenario once more data is uploaded

# start_times = [0.15, 0.45, 0.75, 1.05, 1.35, 1.65, 1.95]
# averages = [0.15, 0.15, 0.15, 0.15]
evoked_left.plot_topomap(times="interactive", ch_type='eeg')
# evoked_left.plot_topomap(times=start_times, average=averages, ch_type='eeg')
evoked_right.plot_topomap(times="interactive", ch_type='eeg')
# evoked_right.plot_topomap(times=start_times, average=averages, ch_type='eeg')
