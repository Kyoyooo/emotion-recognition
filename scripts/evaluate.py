# scripts/evaluate.py
import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from sklearn.metrics import classification_report, confusion_matrix

# Thêm đường dẫn gốc để import từ src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_module import VSMECDataModule

def load_config(config_path="configs/train.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    os.makedirs("results", exist_ok=True)
    
    print("1. Đang tải cấu hình và tập Test qua DataModule...")
    num_labels = config['model']['num_labels']
    labels_list = config['model']['labels_list']
    
    data_module = VSMECDataModule(config['data']['processed_dir'], num_labels)
    # Chỉ lấy test_dataset, bỏ qua train và val
    _, _, test_dataset = data_module.get_datasets()
    
    print("2. Đang tải Mô hình tốt nhất và Tokenizer...")
    model_dir = config['training']['final_model_dir']
    
    if not os.path.exists(model_dir):
        print(f"❌ Lỗi: Không tìm thấy model tại '{model_dir}'. Hãy chạy train.py trước!")
        return

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

    print("3. Thực hiện dự đoán trên tập Test (Blind Set)...")
    trainer = Trainer(model=model)
    predictions_output = trainer.predict(test_dataset)
    
    logits = predictions_output.predictions
    true_labels = predictions_output.label_ids
    
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_labels = np.argmax(logits, axis=-1)
    confidences = np.max(probs, axis=-1)

    print("\n" + "="*50)
    print("BÁO CÁO PHÂN LOẠI (CLASSIFICATION REPORT)")
    print("="*50)
    report = classification_report(true_labels, pred_labels, target_names=labels_list)
    print(report)

    # ---------------------------------------------------------
    # 4. VẼ CONFUSION MATRIX
    # ---------------------------------------------------------
    print("4. Đang vẽ và lưu Confusion Matrix...")
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_list, yticklabels=labels_list)
    plt.title('Confusion Matrix trên tập Test')
    plt.ylabel('Nhãn Thực Tế')
    plt.xlabel('Nhãn Dự Đoán')
    plt.tight_layout()
    
    cm_path = 'results/confusion_matrix.png'
    plt.savefig(cm_path, dpi=300)
    print(f" -> Đã lưu biểu đồ: {cm_path}")

    # ---------------------------------------------------------
    # 5. ERROR ANALYSIS (PHÂN TÍCH LỖI)
    # ---------------------------------------------------------
    print("\n5. Đang trích xuất các trường hợp dự đoán sai (Error Analysis)...")
    errors = []
    
    for i in range(len(true_labels)):
        if true_labels[i] != pred_labels[i]:
            # Dịch ngược token thành chữ để xem text sau khi tiền xử lý
            text = tokenizer.decode(test_dataset[i]['input_ids'], skip_special_tokens=True)
            errors.append({
                'text': text,
                'true_label': labels_list[true_labels[i]],
                'predicted_label': labels_list[pred_labels[i]],
                'confidence': confidences[i]
            })
            
    error_df = pd.DataFrame(errors).sort_values(by='confidence', ascending=False)
    error_csv_path = 'results/error_analysis.csv'
    error_df.to_csv(error_csv_path, index=False, encoding='utf-8-sig')
    
    print(f" -> Đã tìm thấy {len(error_df)} mẫu dự đoán sai.")
    print(f" -> Đã lưu file phân tích lỗi tại: {error_csv_path}")

if __name__ == "__main__":
    main()