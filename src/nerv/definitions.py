import os
from pathlib import Path

import yaml


def load_settings():
    with open(SETTINGS_PATH, "r") as settings_file:
        settings = yaml.load(settings_file, Loader=yaml.FullLoader)
    return settings


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = Path(SRC_DIR)
ROOT_PATH = (SRC_PATH / "../..").resolve()
SETTINGS_PATH = os.path.join(ROOT_PATH, "settings.yaml")
SETTINGS = load_settings()
DATA_PATH = (ROOT_PATH / "{0}".format(SETTINGS["data_dir"])).resolve()
# TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")