from nerv.data.grasp import GraspAndLiftLoader, GraspAndLiftDataset
from nerv.definitions import DATA_PATH, ROOT_PATH
from nerv.utils.frequency import eeg_batch_frequencies

d = DATA_PATH / "grasp-and-lift-eeg-detection"

out_path = (ROOT_PATH / "out")
out_path.mkdir(exist_ok=True)

for dataset in GraspAndLiftLoader(subjects={1}).load(d, n_rows=1000):
    batch_feature_vectors = []
    for batch_idx, batch in enumerate(dataset.rolling_window(250, stride=50)):
        freqs = eeg_batch_frequencies(batch, sampling_rate=GraspAndLiftDataset.sampling_rate)
        freqs.plot(out_path / "frequency_response_{0}_{1}_batch{2}.html".format(dataset.subject, dataset.series, batch_idx))
