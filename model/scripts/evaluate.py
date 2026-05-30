import os
import sys
import json

if sys.path and sys.path[0].endswith('scripts'):
    sys.path.pop(0)

import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_module import EmotionDataModule

def main():
    # 1. Đọc cấu hình từ train.yaml
    with open("configs/train.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    os.makedirs("results", exist_ok=True)
    
    num_labels = config['model']['num_labels']
    labels_list = config['model']['labels_list']
    model_name = config['model']['name']
    
    # 2. Tải tập dữ liệu Test qua DataModule
    data_module = EmotionDataModule(config['data']['processed_dir'], num_labels)
    _, _, test_dataset = data_module.get_datasets()
    
    # 3. Khởi tạo mô hình và Tokenizer từ thư mục lưu trữ
    model_dir = config['training']['final_model_dir']
    if not os.path.exists(model_dir):
        sys.exit(f"❌ Lỗi: Không tìm thấy mô hình đã huấn luyện tại '{model_dir}'. Hãy chạy train trước!")
        
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

    # 4. Thực thi dự đoán trên tập Test
    print(f"📊 Đang tiến hành đánh giá mô hình {model_name} trên tập Test...")
    trainer = Trainer(model=model)
    predictions_output = trainer.predict(test_dataset)
    
    logits = predictions_output.predictions
    true_labels = predictions_output.label_ids
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_labels = np.argmax(logits, axis=-1)
    confidences = np.max(probs, axis=-1)

    # 5. Tính toán các chỉ số Performance chi tiết (4 chữ số thập phân)
    report = classification_report(true_labels, pred_labels, target_names=labels_list, digits=4)
    cm = confusion_matrix(true_labels, pred_labels)

    acc = accuracy_score(true_labels, pred_labels)
    # Vì bài toán mất cân bằng lớp, ta ưu tiên theo dõi cả Macro và Weighted tương tự Classification Report
    precision_macro = precision_score(true_labels, pred_labels, average="macro")
    recall_macro = recall_score(true_labels, pred_labels, average="macro")
    f1_macro = f1_score(true_labels, pred_labels, average="macro")
    
    # Lấy giá trị Loss thực tế thu được trên tập Test từ Trainer
    test_loss = predictions_output.metrics.get("test_loss", 0.0)
    # Tính tổng số lượng tham số (weights) của mô hình RoBERTa
    num_weights = model.num_parameters()

    print("\n" + "="*60)
    print("BÁO CÁO PHÂN LOẠI (CLASSIFICATION REPORT)")
    print("="*60)
    print(report)

    # 6. Vẽ và lưu Confusion Matrix dạng biểu đồ hình ảnh
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_list, yticklabels=labels_list)
    plt.title(f'Confusion Matrix trên tập Test ({model_name})')
    plt.ylabel('Nhãn Thực Tế')
    plt.xlabel('Nhãn Dự Đoán')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300)
    plt.close()

    # 7. Lưu báo cáo chi tiết dưới dạng FILE MARKDOWN (.md)
    md_path = os.path.join("results", "roberta_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {model_name.upper()} Model — Test Results\n\n")
        f.write(f"- **Dataset:** dair-ai/emotion (unsplit split)\n")
        f.write(f"- **Test Loss:** {test_loss:.4f}\n")
        f.write(f"- **Accuracy:** {acc:.4f}\n")
        f.write(f"- **Precision (Macro):** {precision_macro:.4f}\n")
        f.write(f"- **Recall (Macro):** {recall_macro:.4f}\n")
        f.write(f"- **F1-score (Macro):** {f1_macro:.4f}\n")
        f.write(f"- **Weights (Total Parameters):** {num_weights:,}\n\n")
        f.write("## Classification Report\n\n")
        f.write("```\n")
        f.write(report)
        f.write("```\n\n")
        f.write("## Confusion Matrix\n\n")
        f.write("```\n")
        f.write(np.array2string(cm, separator=", "))
        f.write("\n```\n")

    # 8. Lưu báo cáo dạng FILE JSON (.json) để phục vụ tổng hợp Summary sau này
    json_path = os.path.join("results", "roberta_results.json")
    json_data = {
        "model": model_name,
        "dataset": "dair-ai/emotion",
        "weights": num_weights,
        "loss": round(float(test_loss), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision_macro), 4),
        "recall": round(float(recall_macro), 4),
        "f1_score": round(float(f1_macro), 4),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    # 9. Xuất file phân tích các câu bị đoán sai (Error Analysis)
    errors = []
    for i in range(len(true_labels)):
        if true_labels[i] != pred_labels[i]:
            text = tokenizer.decode(test_dataset[i]['input_ids'], skip_special_tokens=True)
            errors.append({
                'text': text, 
                'true_label': labels_list[true_labels[i]],
                'predicted_label': labels_list[pred_labels[i]], 
                'confidence': float(confidences[i])
            })
            
    pd.DataFrame(errors).sort_values(by='confidence', ascending=False).to_csv('results/error_analysis.csv', index=False)
    
    print("-" * 60)
    print(f"✅ Hoàn tất! Tất cả kết quả đã được lưu trữ cấu trúc:")
    print(f"  -> Biểu đồ Confusion Matrix  : results/confusion_matrix.png")
    print(f"  -> File báo cáo Markdown     : {md_path}")
    print(f"  -> File báo cáo cấu trúc JSON: {json_path}")
    print(f"  -> Bảng phân tích lỗi CSV    : results/error_analysis.csv")
    print("=" * 60)

if __name__ == "__main__":
    main()
