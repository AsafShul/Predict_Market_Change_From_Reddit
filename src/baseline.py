#!%PYTHON_HOME%\python.exe
# coding: utf-8

from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax
from sklearn.metrics import f1_score, accuracy_score

from get_dataset import Datasets

ROBERTA_MAX_LENGTH = 512

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"


class RedditStockPredictionBaseline:
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.config = AutoConfig.from_pretrained(MODEL)

        self.model.to(self.device)

        self.datasets = Datasets()

    @staticmethod
    def preprocess(text):
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
        text = ' '.join(('http' if w.startswith('http') else w) for w in text.split(' '))
        return text

    def predict(self, text, return_scores=False):
        text = self.preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors='pt',
                                       truncation=True, padding=True, max_length=ROBERTA_MAX_LENGTH)
        encoded_input = encoded_input.to(self.device)
        output = self.model(**encoded_input)
        scores = output[0][0].cpu().detach().numpy()
        if return_scores:
            return scores

        scores = softmax(scores)

        # ranking = np.argsort(scores)[::-1]

        # for i in range(scores.shape[0]):
        #     l = self.config.id2label[ranking[i]]
        #     s = scores[ranking[i]]
        #     print(f"{i + 1}) {l} {np.round(float(s), 4)}")

        # return ranking[0]

        return scores.argmax()

    def test(self):
        test_set = self.datasets.test_set
        by_day = test_set.groupby(test_set.post_time.dt.date)

        results = pd.DataFrame(columns=['true', 'predicted'])

        for day, df in tqdm(by_day, total=len(by_day), desc="Testing day by day"):
            predictions = np.empty((0, 3))

            labels = df.label.unique()
            assert len(labels) == 1
            label = labels[0]

            for i, row in df.iterrows():
                prediction = self.predict(row.post, return_scores=True)
                predictions = np.vstack((predictions, prediction))

            scores = predictions.mean(axis=0)
            # scores = softmax(scores)
            day_prediction = scores.argmax()

            results.loc[day] = [label, day_prediction]

        score = f1_score(results.true, results.predicted, average='macro')
        accuracy = accuracy_score(results.true, results.predicted)

        return score, accuracy


def main():
    r = RedditStockPredictionBaseline()
    score, accuracy = r.test()
    print("F1 Score:", score)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
