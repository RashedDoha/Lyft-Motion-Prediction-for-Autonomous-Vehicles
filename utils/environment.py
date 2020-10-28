import os
from l5kit.data import LocalDataManager

def set_environ(cfg, root):
    # root directory
    DIR_INPUT = root

    #submission
    SINGLE_MODE_SUBMISSION = f"{DIR_INPUT}/single_mode_sample_submission.csv"
    MULTI_MODE_SUBMISSION = f"{DIR_INPUT}/multi_mode_sample_submission.csv"

    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
    dm = LocalDataManager(None)
    return dm