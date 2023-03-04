import mne
import numpy as np
import pandas as pd
from src.nerv.definitions import DATA_PATH

def convert(microNum):
    return microNum/(10**6)

# subject 1: control- M, 44 yrs old, 16 yrs education
# Read the CSV file as pandas df, then convert to np array
raw_data = pd.read_csv(DATA_PATH / "controls/1.csv", delimiter=',')
eeg_data = raw_data.iloc[:, 4:68].to_numpy().T
convert_eeg = np.vectorize(convert)
eeg_data = convert_eeg(eeg_data)
data = raw_data.iloc[:, 0:4].to_numpy().T
data = np.concatenate((data, eeg_data), axis=0)
# print(eeg_data.shape)  # [sensors, samples]
# print(data.shape)
# channel names
cols = pd.read_csv(DATA_PATH / "columnLabels.csv", delimiter=',')
col_names = list(cols)
ch_names = col_names[4:68]
ch_cond_names = col_names[0:68]
# print(ch_names)
#
# Sampling rate estimate
sfreq = 1024  # Hz

# Create the info structure needed by MNE
eeg_info = mne.create_info(ch_names, sfreq, ch_types='eeg')
info = mne.create_info(ch_cond_names, sfreq)

# create the Raw object
raw_eeg = mne.io.RawArray(eeg_data, eeg_info)
raw = mne.io.RawArray(data, info)

# raw_eeg.plot()

# set up montages (3D locations of electrodes)
montage = mne.channels.make_standard_montage('biosemi64')
raw_eeg.set_montage(montage)
# fig = raw_eeg.plot_sensors(show_names=True)

# set up events
events = mne.find_events(raw, stim_channel='condition', initial_event=True, consecutive=True)
event_dict = {'button tone': 1, 'tone only': 2, 'button only': 3}
# fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'], event_id=event_dict)
# fig.subplots_adjust(right=0.7)  # make room for legend

# raw_eeg.plot(events=events, start=0, duration=10, color='gray', event_color={1: 'r', 2: 'g', 3: 'b'})

# epochs using events from above
epochs = mne.Epochs(raw_eeg, events, event_id=event_dict, tmin=-1.5, tmax=1.5, baseline=None)
# epochs.plot_image(picks=['Fz', 'FCz', 'Cz'])

conditions = ['button tone', 'tone only', 'button only']
epochs.equalize_event_counts(conditions)
button_tone_epochs = epochs['button tone']
tone_epochs = epochs['tone only']
button_epochs = epochs['button only']

# button_tone_epochs.plot_image(picks=['Fz', 'FCz', 'Cz'], title='Button Tone')
# tone_epochs.plot_image(picks=['Fz', 'FCz', 'Cz'], title='Tone')
# button_epochs.plot_image(picks=['Fz', 'FCz', 'Cz'], title='Button')

# subject 69: patient- M, 47 yrs old, 16 yrs education
patient_data = pd.read_csv(DATA_PATH / "controls/1.csv", delimiter=',')
eeg_data = patient_data.iloc[:, 4:68].to_numpy().T
eeg_data = convert_eeg(eeg_data)
raw_data = patient_data.iloc[:, 0:4].to_numpy().T
raw_data = np.concatenate((raw_data, eeg_data), axis=0)

# create the Raw objects
raw_patient_eeg = mne.io.RawArray(eeg_data, eeg_info)
raw_patient = mne.io.RawArray(raw_data, info)

# set up montages (3D locations of electrodes)
raw_patient_eeg.set_montage(montage)

# set up events
patient_events = mne.find_events(raw_patient, stim_channel='condition', initial_event=True, consecutive=True)

# epochs using events from above
patient_epochs = mne.Epochs(raw_patient_eeg, patient_events, event_id=event_dict, tmin=-1.5, tmax=1.5, baseline=None)

patient_epochs.equalize_event_counts(conditions)
patient_button_tone_epochs = patient_epochs['button tone']
patient_tone_epochs = patient_epochs['tone only']
patient_button_epochs = patient_epochs['button only']

# patient_button_tone_epochs.plot_image(picks=['Fz', 'FCz', 'Cz'], title='Patient Button Tone')
# patient_tone_epochs.plot_image(picks=['Fz', 'FCz', 'Cz'], title='Patient Tone')
# patient_button_epochs.plot_image(picks=['Fz', 'FCz', 'Cz'], title='Patient Button')

# make evoked objects
patient_evoked_button_tone = patient_button_tone_epochs.average()
patient_evoked_button = patient_button_epochs.average()
patient_evoked_tone = patient_tone_epochs.average()
control_evoked_button_tone = button_tone_epochs.average()
control_evoked_button = patient_button_epochs.average()
control_evoked_tone = tone_epochs.average()

# mne evoked built-in visualization, combines all sensors in various ways
button_tone_evokes = [patient_evoked_button_tone, control_evoked_button_tone]
for combine in ('mean', 'median', 'gfp', 'std'):
    mne.viz.plot_compare_evokeds(button_tone_evokes, picks='eeg', combine=combine)

tone_evokes = [patient_evoked_tone, control_evoked_tone]
for combine in ('mean', 'median', 'gfp', 'std'):
    mne.viz.plot_compare_evokeds(tone_evokes, picks='eeg', combine=combine)

# use all sensors, choose most important sensors, or combine like above?

# as np array
patient_evoked_button_tone = patient_evoked_button_tone.data.astype('float64')
patient_evoked_button = patient_evoked_button.data.astype('float64')
patient_evoked_tone = patient_evoked_tone.data.astype('float64')
control_evoked_button_tone = control_evoked_button_tone.data.astype('float64')
control_evoked_button = control_evoked_button.data.astype('float64')
control_evoked_tone = control_evoked_tone.data.astype('float64')

# get rid of activity due to only pressing button
patient_button_subtracted = np.subtract(patient_evoked_button_tone, patient_evoked_button)
control_button_subtracted = np.subtract(control_evoked_button_tone, control_evoked_button)

# diff between hearing tone that is self generated by button vs externally generated
patient_suppression = np.subtract(patient_button_subtracted, patient_evoked_tone)
control_suppression = np.subtract(control_button_subtracted, control_evoked_tone)
