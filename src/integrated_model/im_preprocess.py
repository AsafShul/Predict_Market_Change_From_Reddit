#!%PYTHON_HOME%\python.exe
# coding: utf-8

import pandas as pd
import datasets as hf_datasets

from consts import POSTS_PER_DAY, ROBERTA_MAX_LENGTH


class IMPreprocess:
    def __init__(self, tokenizer):
        self.tokenizer_func = self.get_tokenizer_func(tokenizer)
        self.num_comments_mean = None
        self.num_comments_std = None

    @staticmethod
    def get_tokenizer_func(tokenizer):
        def preprocess_function(examples):
            result = tokenizer(examples['text'], truncation=True, padding='max_length',
                               max_length=ROBERTA_MAX_LENGTH)  #, return_tensors='pt')
            return result

        return preprocess_function

    @staticmethod
    def trim_to_posts_with_most_comments_per_day(dataset):
        return dataset.groupby(pd.to_datetime(dataset.post_time).dt.date)\
                      .apply(lambda group: group.sort_values(by='num_comments', ascending=False).head(POSTS_PER_DAY))\
                      .reset_index(drop=True)

    def fit_normalize_comments(self, dataset):
        dataset = self.trim_to_posts_with_most_comments_per_day(dataset)
        self.num_comments_mean = dataset.num_comments.mean()
        self.num_comments_std = dataset.num_comments.std()

    @staticmethod
    def preprocess_text(text):
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
        text = ' '.join(('http' if w.startswith('http') else w) for w in text.split(' '))
        return text

    def preprocess(self, dataset):
        dataset = dataset
        dataset = self.trim_to_posts_with_most_comments_per_day(dataset).copy()

        dataset.post_time = pd.to_datetime(dataset.post_time).dt.date
        dataset.num_comments = (dataset.num_comments - self.num_comments_mean) / self.num_comments_std
        dataset = dataset.rename(columns={'post': 'text'})
        dataset.text = dataset.text.apply(self.preprocess_text)
        dataset.label = dataset.label.astype(int)

        dataset = hf_datasets.Dataset.from_pandas(dataset)
        dataset = dataset.map(self.tokenizer_func, batched=True)
        return dataset
