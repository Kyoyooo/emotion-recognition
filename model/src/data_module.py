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

    def compute_class_weights(self):
        train_labels = self.dataset["train"]["label"]
        class_counts = Counter(train_labels)
        total_samples = len(train_labels)
        
        weights = [total_samples / (self.num_labels * class_counts.get(i, 1)) for i in range(self.num_labels)]
        return torch.tensor(weights, dtype=torch.float)
