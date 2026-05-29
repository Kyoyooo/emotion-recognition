# scripts/inference.py
import os
import sys
import yaml
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.text_cleaner import EnglishTextCleaner

def predict_emotion_calibrated(text, model, tokenizer, cleaner, id2label, thresholds, device):
    cleaned_text = cleaner.clean(text)
    if not cleaned_text.strip(): return "Unknown", 0.0, ""

    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(**inputs).logits, dim=-1).cpu().numpy()[0]
        
    # Áp dụng căn chỉnh ngưỡng quyết định
    calibrated_probs = probs / thresholds
    predicted_id = np.argmax(calibrated_probs)
    
    # Trả về nhãn dự đoán và độ tự tin thực tế (gốc) của nhãn đó
    return id2label[predicted_id], probs[predicted_id], cleaned_text

def main():
    with open("configs/train.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cleaner = EnglishTextCleaner()
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = AutoModelForSequenceClassification.from_pretrained(config['training']['final_model_dir']).to(device).eval()
    
    labels_list = config['model']['labels_list']
    id2label = {i: label for i, label in enumerate(labels_list)}
    
    # Tải bộ ngưỡng tối ưu nếu có
    thresholds_path = "results/thresholds.yaml"
    if os.path.exists(thresholds_path):
        print(" -> Đang cấu hình bộ ngưỡng quyết định tối ưu từ tập Validation...")
        with open(thresholds_path, "r", encoding="utf-8") as f:
            t_dict = yaml.safe_load(f)
        thresholds = np.array([t_dict[label] for label in labels_list])
    else:
        print(" -> Không tìm thấy file thresholds.yaml, tự động dùng ngưỡng mặc định 0.5")
        thresholds = np.ones(len(labels_list)) * 0.5
    
    print("Documents loaded successfully.")
    print("✅ HỆ THỐNG SẴN SÀNG (Nhập 'q' để thoát)")
    while True:
        user_input = input("\n📝 Text: ")
        if user_input.strip().lower() in ['q', 'quit', 'exit']: break
        if not user_input.strip(): continue
            
        label, conf, cleaned = predict_emotion_calibrated(user_input, model, tokenizer, cleaner, id2label, thresholds, device)
        print(f"   [Cleaned]: {cleaned}\n   [Predict]: {label} ({conf*100:.2f}%)")

if __name__ == "__main__":
    main()
