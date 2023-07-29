#!%PYTHON_HOME%\python.exe
# coding: utf-8

import pandas as pd

from consts import DatasetSplitTimes as DST, DATASET_PATH


class Datasets:
    def __init__(self):
        self._full_dataset = None
        self._train_set = None
        self._val_set = None
        self._test_set = None

    @classmethod
    def _get_dataset(cls, df, start, end, tag):
        print(f"Fetching {tag} set from {start} to {end}...")
        filtered = df[(df.post_time >= start) & (df.post_time <= end)].copy()
        # preprocessed = cls.preprocess(filtered)
        return filtered

    @property
    def full_dataset(self):
        if self._full_dataset is None:
            self._full_dataset = pd.read_feather(DATASET_PATH)
        return self._full_dataset

    @property
    def train_set(self):
        if self._train_set is None:
            self._train_set = self._get_dataset(self.full_dataset, DST.TRAIN_START, DST.VAL_START, 'train')
        return self._train_set

    @property
    def val_set(self):
        if self._val_set is None:
            self._val_set = self._get_dataset(self.full_dataset, DST.VAL_START, DST.TEST_START, 'val')
        return self._val_set

    @property
    def test_set(self):
        if self._test_set is None:
            self._test_set = self._get_dataset(self.full_dataset, DST.TEST_START, DST.TEST_END, 'test')
        return self._test_set


if __name__ == '__main__':
    datasets = Datasets()
    print("Full dataset:", datasets.full_dataset.shape[0])
    print("Train set:", datasets.train_set.shape[0])
    print("Val set:", datasets.val_set.shape[0])
    print("Test set:", datasets.test_set.shape[0])

