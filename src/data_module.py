import torch
from collections import Counter
from datasets import load_from_disk

class VSMECDataModule:
    def __init__(self, processed_dir, num_labels):
        self.processed_dir = processed_dir
        self.num_labels = num_labels
        self.dataset = load_from_disk(processed_dir)

    def get_datasets(self):
        """Trả về tuple: (train_dataset, val_dataset, test_dataset)"""
        return self.dataset["train"], self.dataset["validation"], self.dataset["test"]

    def compute_class_weights(self):
        """Tính toán trọng số nhãn cho hàm Weighted Loss"""
        train_labels = self.dataset["train"]["label"]
        class_counts = Counter(train_labels)
        total_samples = len(train_labels)
        
        # Thêm .get(i, 1) để tránh lỗi chia cho 0 nếu tập train khuyết 1 nhãn nào đó
        weights = [total_samples / (self.num_labels * class_counts.get(i, 1)) for i in range(self.num_labels)]
        return torch.tensor(weights, dtype=torch.float)