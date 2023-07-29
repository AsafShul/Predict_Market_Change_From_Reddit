#!%PYTHON_HOME%\python.exe
# coding: utf-8

import torch
from torch import nn
from torch.nn.functional import one_hot

from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from consts import POSTS_PER_DAY, FC_INPUT_DIM, FC_OUTPUT_DIM, BASE_MODEL


class IntegratedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, cache_dir="./cache")
        self.fc = nn.Sequential(
            nn.Linear(FC_INPUT_DIM, FC_OUTPUT_DIM),
            nn.Linear(FC_OUTPUT_DIM, 3)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, num_comments, labels):
        batches = [
            dict(
                input_ids=input_ids[:, i],
                attention_mask=attention_mask[:, i],
            )
            for i in range(POSTS_PER_DAY)
        ]

        model_predictions = torch.hstack([self.model(**sub_batch).logits for sub_batch in batches])
        preds_and_comments = torch.hstack([model_predictions, num_comments])
        # preds_and_comments = torch.Tensor([i for seq, com in zip(model_predictions, comments) for i in (seq, com)])

        output = self.fc(preds_and_comments)

        loss = self.criterion(output, one_hot(labels, 3).float())
        return SequenceClassifierOutput(
            loss=loss,
            logits=output
        )
