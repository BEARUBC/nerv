import mne

from src.nerv.data.dataset import EEGDataset


class EEGVisualizer:
    def __init__(self, **settings):
        pass

    def visualize(self, dtst: EEGDataset):
        # MNE analysis with epochs plots
        # epochs.average().plot()
        # epochs.plot_psd()

        mne_dtst = dtst.dataset
        evoked_left = mne_dtst['Left_Hand'].average()
        evoked_right = mne_dtst['Right_Hand'].average()
        evokeds = dict(Left=evoked_left, Right=evoked_right)
        picks1 = 'C3'
        mne.viz.plot_compare_evokeds(evokeds, picks=picks1, colors=['r', 'b'])

        picks2 = 'C4'
        mne.viz.plot_compare_evokeds(evokeds, picks=picks2, colors=['r', 'b'])
        # Experiment used grand-averaged brain response, which likely is the average across all subjects
        # mne.grand_average() should be used in this scenario once more data is uploaded

        # start_times = [0.15, 0.45, 0.75, 1.05, 1.35, 1.65, 1.95]
        start_times = [0.45, 1.25, 2.05, 2.85]
        averages = [0.302, 0.502, 0.502, 0.502]
        # evoked_left.plot_topomap(times="interactive", ch_type='eeg')
        evoked_left.plot_topomap(times=start_times, average=averages, ch_type='eeg')
        # evoked_right.plot_topomap(times="interactive", ch_type='eeg')
        evoked_right.plot_topomap(times=start_times, average=averages, ch_type='eeg')
