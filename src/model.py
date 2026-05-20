import sys
import os
# Hack để tránh lỗi "module 'evaluate' has no attribute 'load'"
if sys.path and sys.path[0].endswith('scripts'):
    sys.path.pop(0)

import torch
import torch.nn as nn
import numpy as np
from transformers import Trainer
import evaluate

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    logits, true_labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_metric.compute(predictions=predictions, references=true_labels)
    f1 = f1_metric.compute(predictions=predictions, references=true_labels, average="macro")
    
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}