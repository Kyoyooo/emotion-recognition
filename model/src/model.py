# src/model.py
import sys
import os
if sys.path and sys.path[0].endswith('scripts'):
    sys.path.pop(0)

import torch
import torch.nn as nn
import numpy as np
from transformers import Trainer
import evaluate

class FocalLoss(nn.Module):
    """
    Triển khai Focal Loss chuẩn cho phân loại đa lớp có tích hợp Class Weights.
    FL(p_t) = -w_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        # Đăng ký weight như một buffer để PyTorch tự động quản lý thiết bị phần cứng (CPU/GPU)
        self.register_buffer('weight', weight)
        # Sử dụng reduction='none' để lấy loss thô của từng mẫu trước khi nhân hệ số focal
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)  # pt chính là xác suất dự đoán đúng của mẫu đó (p_t)
        
        # Tính toán Focal Loss thô
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Áp dụng trọng số lớp (Class Weights) nếu có cấu hình
        if self.weight is not None:
            sample_weights = self.weight[targets]
            focal_loss = focal_loss * sample_weights
            
        return focal_loss.mean()

class CustomLossTrainer(Trainer):
    def __init__(self, class_weights, loss_type="weighted_ce", gamma=2.0, label_smoothing=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_type = loss_type
        self.gamma = gamma
        self.label_smoothing = label_smoothing # Nhận hệ số làm mịn nhãn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.loss_type == "focal_loss":
            loss_fct = FocalLoss(weight=self.class_weights.to(model.device), gamma=self.gamma)
        else:
            # Tích hợp native label_smoothing của PyTorch vào CrossEntropyLoss
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights.to(model.device),
                label_smoothing=self.label_smoothing
            )
            
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
