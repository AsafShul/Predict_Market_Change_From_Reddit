import os
import datetime as dt
import json
import wandb
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from evaluate import load
from sklearn.metrics import f1_score, accuracy_score

from get_dataset import Datasets

import datasets

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    EvalPrediction,
    TrainingArguments
)

ROBERTA_MAX_LENGTH = 512

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
WANDB_DIR = os.path.join('..', 'wandb-logs')
WANDB_API_KEY = 'b7810f968d2bea395c268dab82307d9e5d443533'
CHECKPOINT_DIR = "../results/second_run/checkpoint-220620/"


class RedditStockPredictionFinetune:
    def __init__(self):
        self.device = torch.device('cuda:0')
        self.model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR, cache_dir="./cache").to(self.device)
        # self.model = AutoModelForSequenceClassification.from_pretrained(MODEL, cache_dir="./cache").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.config = AutoConfig.from_pretrained(MODEL)
        self.datasets = Datasets()
        self.pre_process_func = self.get_tokenizer_func(self.tokenizer)
        self.output_dir = os.path.join('..', 'results', 'second_run')
        self.training_args = TrainingArguments(output_dir=self.output_dir,
                                               evaluation_strategy="epoch",
                                               save_strategy="epoch",
                                               optim="adamw_torch",
                                               num_train_epochs=12)

    @staticmethod
    def preprocess(text):
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
        text = ' '.join(('http' if w.startswith('http') else w) for w in text.split(' '))
        return text

    @staticmethod
    def get_metric_func(metric_name):
        metric = load(metric_name)

        def compute_metrics(p: EvalPrediction):
            preds = np.argmax(p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions, axis=1)
            return metric.compute(predictions=preds, references=p.label_ids)
            # result = metric.compute(predictions=preds, references=p.label_ids)[metric_name]
            # return {metric_name: result}

        return compute_metrics

    def get_tokenizer_func(self, tokenizer):
        def preprocess_function(examples):
            result = tokenizer(examples['text'], truncation=True,
                               max_length=ROBERTA_MAX_LENGTH)  #, return_tensors='pt')
            return result

        return preprocess_function

    def train(self):
        wandb.init(project="reddit-stock-prediction",
                   name=f"reddit-stock-prediction_{dt.datetime.now()}",
                   dir=os.path.join('..', 'wandb-logs'))
        train_set = self.datasets.train_set.rename(columns={'post': 'text'})
        val_set = self.datasets.val_set.rename(columns={'post': 'text'})

        train_set.text = train_set.text.apply(self.preprocess)
        val_set.text = val_set.text.apply(self.preprocess)

        train_set.label = train_set.label.astype(int)
        val_set.label = val_set.label.astype(int)

        train_set = datasets.Dataset.from_pandas(train_set)
        val_set = datasets.Dataset.from_pandas(val_set)

        train_set = train_set.map(self.pre_process_func, batched=True)
        val_set = val_set.map(self.pre_process_func, batched=True)

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_set,
            eval_dataset=val_set,
            compute_metrics=self.get_metric_func('accuracy'),
            tokenizer=self.tokenizer,
        )

        train_result = trainer.train()
        trainer.save_model(f'models/{MODEL}')
        wandb.finish()

        return train_result

    def test(self):
        test_set = self.datasets.test_set.iloc[:1000]
        test_set = test_set.rename(columns={'post': 'text'})
        test_set = test_set.set_index(pd.to_datetime(test_set.post_time))
        test_set.text = test_set.text.apply(self.preprocess)
        test_set.label = test_set.label.astype(int)

        by_day = test_set.groupby(test_set.index.date)
        results = pd.DataFrame(columns=['true', 'predicted'])

        for day, df in tqdm(by_day, total=len(by_day), desc="Testing day by day"):
            predictions = np.empty((0, 3))

            labels = df.label.unique()
            assert len(labels) == 1
            label = labels[0]

            for i, row in df.iterrows():
                encoded_input = self.tokenizer(row.text, return_tensors='pt',
                                               truncation=True, padding=True, max_length=ROBERTA_MAX_LENGTH).to(self.device)
                prediction = self.model(**encoded_input)
                predictions = np.vstack((predictions, prediction.logits.detach().cpu().numpy()))

            scores = predictions.mean(axis=0)
            # scores = softmax(scores)
            day_prediction = scores.argmax()

            results.loc[day] = [label, day_prediction]

        print(type(results.true))
        print(type(results.predicted))
        print("\n")
        print(results.true)
        print(results.predicted)
        print("\n")
        print(type(results.true[0]))
        print(type(results.predicted[0]))


        score = f1_score(results.true.to_numpy(), results.predicted.to_numpy(), average='macro')
        accuracy = accuracy_score(results.true.to_numpy(), results.predicted.to_numpy())

        results = dict(score=score,
                       accuracy=accuracy)

        with open(os.path.join(self.output_dir, f'results_{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'),
                  'w') as f:
            json.dump(results, f)

        return score, accuracy


def main():
    os.environ['WANDB_DIR'] = WANDB_DIR
    os.environ['WANDB_CACHE_DIR'] = WANDB_DIR
    os.environ['WANDB_CONFIG_DIR'] = WANDB_DIR
    wandb.login(key=WANDB_API_KEY)

    r = RedditStockPredictionFinetune()
    # train_results = r.train()
    #
    # print('train_results:')
    # print(train_results)

    f1, accuracy = r.test()

    print("F1 Score:", f1)
    print("Accuracy Score:", accuracy)


if __name__ == "__main__":
    main()
