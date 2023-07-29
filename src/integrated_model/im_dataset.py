#!%PYTHON_HOME%\python.exe
# coding: utf-8

import numpy as np
import torch
from torch.utils.data import Dataset
from consts import POSTS_PER_DAY, ROBERTA_MAX_LENGTH


EMPTY_POSTS = [[0 for _ in range(ROBERTA_MAX_LENGTH)] for _ in range(POSTS_PER_DAY)]
EMPTY_COMMENTS = [0 for _ in range(POSTS_PER_DAY)]
EMPTY_LABELS = [1 for _ in range(POSTS_PER_DAY)]


class IMDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self._dates = sorted(set(self.dataset['post_time']))

        self._date_idx_to_post_idxs = []
        self._date_idx_to_num_to_add = []
        self._date_idx_to_label = []

        for i, date in enumerate(self._dates):
            post_idxs = np.where(np.array(self.dataset['post_time']) == date)[0]
            num_posts = len(post_idxs)
            assert num_posts <= POSTS_PER_DAY, f"date {date} has {num_posts} posts > {POSTS_PER_DAY}"
            self._date_idx_to_post_idxs.append(post_idxs)
            self._date_idx_to_num_to_add.append(POSTS_PER_DAY - num_posts)

            labels = set(self.dataset[post_idxs]['label'])
            assert len(labels) == 1, f"date {date} has more than one label: {labels}"
            self._date_idx_to_label.append(labels.pop())

        self.dataset = self.dataset.remove_columns(['post_time', 'text'])

    def __getitem__(self, idx):
        item = self.dataset[self._date_idx_to_post_idxs[idx]]
        label = torch.tensor(self._date_idx_to_label[idx])

        num_to_add = self._date_idx_to_num_to_add[idx]
        permutation = np.random.permutation(POSTS_PER_DAY)

        item['input_ids'] = (torch.tensor(item['input_ids'] + EMPTY_POSTS[:num_to_add]))[permutation]
        item['attention_mask'] = (torch.tensor(item['attention_mask'] + EMPTY_POSTS[:num_to_add]))[permutation]
        item['num_comments'] = (torch.tensor(item['num_comments'] + EMPTY_COMMENTS[:num_to_add]))[permutation]
        item['label'] = label

        return item

    def __len__(self):
        return len(self._dates)
