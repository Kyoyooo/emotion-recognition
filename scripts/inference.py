# scripts/inference.py
import os
import sys
import yaml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Import bộ tiền xử lý đã xây dựng ở Giai đoạn 1
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.text_cleaner import VietnameseTextCleaner

def load_config(config_path="configs/train.yaml"):
    """Tái sử dụng file cấu hình train để lấy đường dẫn model và danh sách nhãn"""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def predict_emotion(text, model, tokenizer, cleaner, id2label, device):
    """
    Hàm thực hiện luồng suy luận (Inference Pipeline)
    """
    # 1. Đưa text thô qua bộ Tiền xử lý sâu (Xóa nhiễu, dịch emoji, chuẩn hóa teencode)
    cleaned_text = cleaner.clean(text)
    
    # Nếu câu sau khi làm sạch bị rỗng (người dùng nhập toàn ký tự đặc biệt)
    if not cleaned_text.strip():
        return "Unknown", 0.0, ""

    # 2. Tokenize dữ liệu (Mã hóa văn bản thành tensor)
    inputs = tokenizer(
        cleaned_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=128
    )
    
    # Chuyển dữ liệu lên cùng device với model (CPU/GPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 3. Chạy model dự đoán
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Áp dụng Softmax để chuyển logit thành xác suất (probability) từ 0 đến 1
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
    # Lấy xác suất cao nhất và id của nhãn đó
    confidence, predicted_id = torch.max(probs, dim=-1)
    
    predicted_label = id2label[predicted_id.item()]
    
    return predicted_label, confidence.item(), cleaned_text

def main():
    print("Đang khởi tạo hệ thống nhận diện cảm xúc...")
    config = load_config()
    
    # 1. Khởi tạo device (Ưu tiên GPU nếu có)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" -> Đang sử dụng thiết bị: {device}")
    
    # 2. Khởi tạo Text Cleaner
    cleaner = VietnameseTextCleaner()
    
    # 3. Tải Tokenizer và Model
    model_dir = config['training']['final_model_dir']
    
    if not os.path.exists(model_dir):
        print(f"❌ Lỗi: Không tìm thấy model tại '{model_dir}'. Bạn cần chạy train.py trước!")
        return

    print(" -> Đang tải Model và Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval() # Chuyển model sang chế độ đánh giá (tắt Dropout)
    
    # 4. Khởi tạo từ điển mapping nhãn
    labels_list = config['model']['labels_list']
    id2label = {i: label for i, label in enumerate(labels_list)}
    
    print("\n✅ HỆ THỐNG ĐÃ SẴN SÀNG!")
    print("-" * 50)
    print("Nhập một câu bình luận để kiểm tra (Gõ 'q' hoặc 'quit' để thoát).")
    
    # Vòng lặp tương tác với người dùng
    while True:
        user_input = input("\n📝 Nhập văn bản: ")
        
        if user_input.strip().lower() in ['q', 'quit', 'exit']:
            print("Đóng hệ thống. Tạm biệt!")
            break
            
        if not user_input.strip():
            continue
            
        # Gọi hàm dự đoán
        label, confidence, cleaned_text = predict_emotion(
            user_input, model, tokenizer, cleaner, id2label, device
        )
        
        # In kết quả
        print(f"   [Tiền xử lý]: {cleaned_text}")
        print(f"   [Dự đoán]   : {label} (Độ tự tin: {confidence*100:.2f}%)")

if __name__ == "__main__":
    main()