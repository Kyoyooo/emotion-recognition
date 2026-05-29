import os
import sys

if sys.path and sys.path[0].endswith('scripts'):
    sys.path.pop(0)

import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_module import EmotionDataModule

def main():
    with open("configs/train.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    os.makedirs("results", exist_ok=True)
    
    num_labels = config['model']['num_labels']
    labels_list = config['model']['labels_list']
    
    data_module = EmotionDataModule(config['data']['processed_dir'], num_labels)
    _, _, test_dataset = data_module.get_datasets()
    
    model_dir = config['training']['final_model_dir']
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

    trainer = Trainer(model=model)
    predictions_output = trainer.predict(test_dataset)
    
    logits = predictions_output.predictions
    true_labels = predictions_output.label_ids
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_labels = np.argmax(logits, axis=-1)
    confidences = np.max(probs, axis=-1)

    print(classification_report(true_labels, pred_labels, target_names=labels_list, digits=4))

    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_list, yticklabels=labels_list)
    plt.savefig('results/confusion_matrix.png', dpi=300)
    
    errors = []
    for i in range(len(true_labels)):
        if true_labels[i] != pred_labels[i]:
            text = tokenizer.decode(test_dataset[i]['input_ids'], skip_special_tokens=True)
            errors.append({
                'text': text, 'true_label': labels_list[true_labels[i]],
                'predicted_label': labels_list[pred_labels[i]], 'confidence': confidences[i]
            })
            
    pd.DataFrame(errors).sort_values(by='confidence', ascending=False).to_csv('results/error_analysis.csv', index=False)
    print("✅ Đã lưu Confusion Matrix và Error Analysis vào thư mục results/")

if __name__ == "__main__":
    main()
