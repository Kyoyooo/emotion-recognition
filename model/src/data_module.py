# src/data_module.py
import torch
from collections import Counter
from datasets import load_from_disk

class EmotionDataModule:
    def __init__(self, processed_dir, num_labels):
        self.processed_dir = processed_dir
        self.num_labels = num_labels
        self.dataset = load_from_disk(processed_dir)

    def get_datasets(self):
        return self.dataset["train"], self.dataset["validation"], self.dataset["test"]

    def compute_class_weights(self, smoothing_alpha=1.0):
        """
        Tính toán trọng số nhãn có áp dụng hệ số làm dịu alpha.
        alpha = 0.5 sẽ chuyển bài toán sang dạng Square Root Smoothing.
        """
        train_labels = self.dataset["train"]["label"]
        class_counts = Counter(train_labels)
        total_samples = len(train_labels)
        
        # 1. Tính trọng số gốc nghịch đảo tuyến tính
        base_weights = [total_samples / (self.num_labels * class_counts.get(i, 1)) for i in range(self.num_labels)]
        
        # 2. Áp dụng smoothing bằng lũy thừa alpha
        smoothed_weights = [w ** smoothing_alpha for w in base_weights]
        
        return torch.tensor(smoothed_weights, dtype=torch.float)
