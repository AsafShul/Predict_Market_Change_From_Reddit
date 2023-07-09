import os
import wandb
import numpy as np
import pandas as pd
import datetime as dt

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


class RedditStockPredictionFinetune:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL, cache_dir="./cache")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.config = AutoConfig.from_pretrained(MODEL)
        self.datasets = Datasets()
        self.pre_process_func = self.get_tokenizer_func(self.tokenizer)
        self.training_args = TrainingArguments(output_dir="results",
                                               evaluation_strategy="epoch",
                                               save_strategy="epoch")

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
            result = metric.compute(predictions=preds, references=p.label_ids)[metric_name]
            return {metric_name: result}

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
        test_score = self.test()
        wandb.finish()

        return train_result, test_score

    def test(self):
        test_set = self.datasets.test_set.rename(columns={'post': 'text'})
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
                                               truncation=True, padding=True, max_length=ROBERTA_MAX_LENGTH)
                prediction = self.model(**encoded_input)
                predictions = np.vstack((predictions, prediction.logits.detach().numpy()))

            scores = predictions.mean(axis=0)
            # scores = softmax(scores)
            day_prediction = scores.argmax()

            results.loc[day] = [label, day_prediction]

        score = f1_score(results.true, results.predicted, average='macro')
        accuracy = accuracy_score(results.true, results.predicted)
        return score, accuracy


def main():
    wandb.login(key=os.environ["WANDB_API_KEY"] if "WANDB_API_KEY" in os.environ else None)
    r = RedditStockPredictionFinetune()
    train_results, test_score = r.train()
    print('train_results:')
    print(train_results)

    f1, accuracy = r.test()
    print("F1 Score:", f1)
    print("Accuracy Score:", accuracy)


if __name__ == "__main__":
    main()
