# src/model.py
import torch
import torch.nn as nn
import numpy as np
from transformers import Trainer
import evaluate

class WeightedTrainer(Trainer):
    """
    Custom Trainer ghi đè hàm compute_loss để áp dụng Weighted Cross-Entropy,
    giúp mô hình tập trung học các nhãn thiểu số (như Surprise, Fear).
    """
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Lấy nhãn thực tế
        labels = inputs.pop("labels")
        
        # Dự đoán
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Hàm Loss với trọng số (đưa weight vào cùng device với model)
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        
        # Tính loss
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    """
    Tính toán Accuracy và Macro F1 để đánh giá model trong quá trình train.
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    logits, true_labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_metric.compute(predictions=predictions, references=true_labels)
    f1 = f1_metric.compute(predictions=predictions, references=true_labels, average="macro")
    
    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"]
    }