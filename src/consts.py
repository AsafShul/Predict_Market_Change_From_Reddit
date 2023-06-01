#!%PYTHON_HOME%\python.exe
# coding: utf-8

import os
from datetime import datetime


class DatasetSplitTimes:
    TRAIN_START = datetime(2018, 1, 1)
    VAL_START = datetime(2021, 7, 1)
    TEST_START = datetime(2022, 1, 1)
    TEST_END = datetime(2022, 12, 31)


DATA_DIR = os.path.join('..',
                        'data')
DATASET_PATH = os.path.join(DATA_DIR,
                            'formatted_df.ftr.zstd')
