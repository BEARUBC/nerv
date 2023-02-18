from src.nerv.data.dataset import EEGDataset
from src.nerv.data.matlab import MatlabLoader
from src.nerv.definitions import DATA_PATH
from src.nerv.visualization.base import EEGVisualizer

loader = MatlabLoader()
subj_path = (DATA_PATH / "test_data").resolve()
print(subj_path)
dtst: EEGDataset = loader.load(subj_path)
visualizer = EEGVisualizer()
visualizer.visualize(dtst)
