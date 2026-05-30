# scripts/evaluate.py
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_module import EmotionDataModule

def find_best_thresholds(val_probs, val_labels, num_labels):
    """
    Tìm kiếm ngưỡng tối ưu cho từng lớp dựa trên tập Validation
    bằng cách tối ưu hóa hàm Macro F1 độc lập cho từng lớp (One-vs-Rest).
    """
    print("   -> Đang quét tìm ngưỡng quyết định tối ưu trên tập Validation...")
    best_thresholds = np.ones(num_labels) * 0.5 # Mặc định ban đầu là 0.5
    
    for i in range(num_labels):
        best_f1 = 0
        best_t = 0.5
        # Quét thử nghiệm các ngưỡng từ 0.1 đến 0.9
        for t in np.linspace(0.1, 0.9, 81):
            # Thử nghiệm cách scale xác suất của lớp i dựa trên ngưỡng t
            scaled_probs = val_probs.copy()
            scaled_probs[:, i] = scaled_probs[:, i] / t
            preds = np.argmax(scaled_probs, axis=-1)
            
            # Tính toán Macro F1 thử nghiệm
            current_f1 = f1_score(val_labels, preds, average='macro')
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_t = t
        best_thresholds[i] = best_t
    return best_thresholds

def main():
    with open("configs/train.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    os.makedirs("results", exist_ok=True)
    
    num_labels = config['model']['num_labels']
    labels_list = config['model']['labels_list']
    
    # 1. Load cả Validation và Test datasets
    data_module = EmotionDataModule(config['data']['processed_dir'], num_labels)
    _, val_dataset, test_dataset = data_module.get_datasets()
    
    model_dir = config['training']['final_model_dir']
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    trainer = Trainer(model=model)

    # 2. Dự đoán trên tập Validation để tìm Ngưỡng
    print("2. Chạy dự đoán trên tập Validation...")
    val_output = trainer.predict(val_dataset)
    val_probs = torch.nn.functional.softmax(torch.tensor(val_output.predictions), dim=-1).numpy()
    val_labels = val_output.label_ids
    
    best_thresholds = find_best_thresholds(val_probs, val_labels, num_labels)
    
    # Lưu bộ ngưỡng vào kết quả để Inference tái sử dụng
    thresholds_dict = {labels_list[i]: float(best_thresholds[i]) for i in range(num_labels)}
    with open("results/thresholds.yaml", "w", encoding="utf-8") as f:
        yaml.dump(thresholds_dict, f)
        
    print("\n📊 BỘ NGƯỠNG TỐI ƯU TÌM ĐƯỢC:")
    for label, t in thresholds_dict.items():
        print(f"   - Ngưỡng lớp {label:<10}: {t:.4f}")

    # 3. Dự đoán trên tập Test để đánh giá thực tế kết quả sau hiệu chuẩn
    print("\n3. Chạy dự đoán hiệu chuẩn (Calibrated) trên tập Test...")
    test_output = trainer.predict(test_dataset)
    test_probs = torch.nn.functional.softmax(torch.tensor(test_output.predictions), dim=-1).numpy()
    test_labels = test_output.label_ids
    
    # Áp dụng công thức điều hòa ngưỡng quyết định (Calibrated Argmax)
    calibrated_probs = test_probs / best_thresholds
    pred_labels = np.argmax(calibrated_probs, axis=-1)
    confidences = np.max(test_probs, axis=-1) # Vẫn giữ độ tự tin gốc để phân tích lỗi

    print("\n" + "="*60)
    print("BÁO CÁO PHÂN LOẠI SAU KHI TINH CHỈNH NGƯỠNG (4 CHỮ SỐ THẬP PHÂN)")
    print("="*60)
    print(classification_report(test_labels, pred_labels, target_names=labels_list, digits=4))

    # 4. Vẽ Confusion Matrix mới
    cm = confusion_matrix(test_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_list, yticklabels=labels_list)
    plt.title('Calibrated Confusion Matrix trên tập Test')
    plt.savefig('results/confusion_matrix.png', dpi=300)
    
    # 5. Xuất file Error Analysis mới
    errors = []
    for i in range(len(test_labels)):
        if test_labels[i] != pred_labels[i]:
            text = tokenizer.decode(test_dataset[i]['input_ids'], skip_special_tokens=True)
            errors.append({
                'text': text, 'true_label': labels_list[test_labels[i]],
                'predicted_label': labels_list[pred_labels[i]], 'confidence': confidences[i]
            })
            
    pd.DataFrame(errors).sort_values(by='confidence', ascending=False).to_csv('results/error_analysis.csv', index=False)
    print("✅ Đã lưu kết quả tối ưu vào thư mục results/")

if __name__ == "__main__":
    main()
