#!%PYTHON_HOME%\python.exe
# coding: utf-8

import torch
from torch import nn

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    EvalPrediction,
    TrainingArguments
)

from consts import POSTS_PER_DAY, FC_INPUT_DIM, FC_OUTPUT_DIM, BASE_MODEL


class IntegratedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, cache_dir="./cache")
        self.fc = nn.Sequential(
            nn.Linear(FC_INPUT_DIM, FC_OUTPUT_DIM),
            nn.Linear(FC_OUTPUT_DIM, 3)
        )

    def forward(self, batch):
        sequences = batch['sequences']
        comments = batch['comments']

        model_predictions = torch.Tensor([self.model(sequence) for sequence in sequences])
        preds_and_comments = torch.hstack([model_predictions, comments])
        # preds_and_comments = torch.Tensor([i for seq, com in zip(model_predictions, comments) for i in (seq, com)])

        output = self.fc(preds_and_comments)
        return output

