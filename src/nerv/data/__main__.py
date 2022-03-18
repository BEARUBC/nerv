from nerv.data.grasp import GraspAndLiftLoader
from nerv.definitions import DATA_PATH

d = DATA_PATH / "grasp-and-lift-eeg-detection"

dtst = GraspAndLiftLoader().load(d, n_rows=10000, subjects=[1])
print("asd")
