import os
import datetime as dt
from tqdm import tqdm
import json

import numpy as np
import pandas as pd
import torch
import wandb

from evaluate import load
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    EvalPrediction,
    TrainingArguments
)

from get_dataset import Datasets
from consts import BASE_MODEL
from integrated_model import IntegratedModel
from im_preprocess import IMPreprocess
from im_dataset import IMDataset


WANDB_DIR = os.path.join('..', '..', 'wandb-logs')
WANDB_API_KEY = 'b7810f968d2bea395c268dab82307d9e5d443533'
BATCH_SIZE = 32


class RedditStockPredictionTraining:
    def __init__(self):
        self.device = torch.device('cuda:0')
        print("Loading model...")
        self.model = IntegratedModel().to(self.device)

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        config = AutoConfig.from_pretrained(BASE_MODEL)
        self.id2label = config.id2label

        self.datasets = Datasets()
        self.preprocessor = IMPreprocess(tokenizer)

        print("Fitting normalization info on num_comments column...")
        self.preprocessor.fit_normalize_comments(self.datasets.train_set)
        print("Preprocessing train set...")
        self.train_set = IMDataset(self.preprocessor.preprocess(self.datasets.train_set))
        print("\rPreprocessing val set...")
        self.val_set = IMDataset(self.preprocessor.preprocess(self.datasets.val_set))
        print("\rPreprocessing test set...")
        self.test_set = IMDataset(self.preprocessor.preprocess(self.datasets.test_set))
        print("\rDone preprocessing.")

        self.output_dir = os.path.join('..', '..', 'results', 'third_run')

        self.training_args = TrainingArguments(output_dir=self.output_dir,
                                               evaluation_strategy="epoch",
                                               save_strategy="no",
                                               optim="adamw_torch",
                                               num_train_epochs=50,
                                               per_device_train_batch_size=2,
                                               per_device_eval_batch_size=2,
                                               report_to="wandb")
        self.trainer = None

    @staticmethod
    def get_metric_func(metric_name):
        metric = load(metric_name)

        def compute_metrics(p: EvalPrediction):
            preds = np.argmax(p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions, axis=1)
            return metric.compute(predictions=preds, references=p.label_ids)
            # result = metric.compute(predictions=preds, references=p.label_ids)[metric_name]
            # return {metric_name: result}

        return compute_metrics

    def train(self):
        print("Training...")

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_set,
            eval_dataset=self.val_set,
            compute_metrics=self.get_metric_func('accuracy'),
        )

        train_result = self.trainer.train()
        # self.trainer.save_model(os.path.join('..', '..', 'models', 'integrated_model'))

        return train_result

    def test(self):
        print("Testing...")

        outputs = self.trainer.predict(self.test_set)

        predictions = outputs.predictions.argmax(axis=1)
        labels = outputs.label_ids

        score = f1_score(labels, predictions, average='macro')
        accuracy = accuracy_score(labels, predictions)

        results = dict(score=score,
                       accuracy=accuracy)

        with open(os.path.join(self.output_dir, f'results_{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'), 'w') as f:
            json.dump(results, f)

        return score, accuracy


def main():
    os.environ.update(dict(WANDB_DIR=WANDB_DIR,
                           WANDB_CACHE_DIR=WANDB_DIR,
                           WANDB_CONFIG_DIR=WANDB_DIR))
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project="reddit-stock-prediction",
               name=f"reddit-stock-prediction_{dt.datetime.now()}",
               dir=os.path.join('../..', 'wandb-logs'))

    r = RedditStockPredictionTraining()
    train_results = r.train()

    print('train_results:')
    print(train_results)

    f1, accuracy = r.test()

    print("F1 Score:", f1)
    print("Accuracy Score:", accuracy)

    wandb.finish()


if __name__ == "__main__":
    main()
