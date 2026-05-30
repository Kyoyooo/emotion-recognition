import os
import sys
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login

def load_config(config_path="configs/train.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def main():
    print("🚀 CHUẨN BỊ UPLOAD MÔ HÌNH LÊN HUGGING FACE")
    print("-" * 50)
    
    # 1. Đọc đường dẫn mô hình từ file cấu hình
    config = load_config()
    model_dir = config['training']['final_model_dir']
    base_model_name = config['model']['name']
    
    if not os.path.exists(model_dir):
        print(f"❌ Lỗi: Không tìm thấy model tại '{model_dir}'.")
        return

    # 2. Yêu cầu nhập thông tin xác thực
    hf_token = input("🔑 Nhập Hugging Face Access Token (loại Write): ").strip()
    repo_id = input("📦 Nhập tên Repository muốn tạo (VD: username/roberta-emotion-english): ").strip()

    if not hf_token or not repo_id:
        print("❌ Lỗi: Token và Tên Repo không được để trống!")
        return

    # 3. Đăng nhập vào Hugging Face Hub
    print("\nĐang xác thực với Hugging Face...")
    login(token=hf_token)

    # 4. Load Model và Tokenizer từ Local
    print("\n1. Đang tải Model và Tokenizer từ ổ cứng...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # 5. Push lên Hub
    print(f"\n2. Đang tiến hành upload lên '{repo_id}' (Quá trình này có thể mất vài phút)...")
    
    # Đẩy model
    model.push_to_hub(
        repo_id, 
        commit_message="Upload fine-tuned RoBERTa for emotion recognition",
        private=False # Đổi thành True nếu bạn muốn ẩn mô hình không cho người khác thấy
    )
    
    # Đẩy tokenizer
    tokenizer.push_to_hub(
        repo_id, 
        commit_message="Upload tokenizer"
    )

    print("\n" + "=" * 50)
    print(f"✅ HOÀN TẤT! Mô hình của bạn đã được đưa lên Cloud.")
    print(f"🌍 Link truy cập: https://huggingface.co/{repo_id}")
    print("=" * 50)
    
    print("\n💡 Cách sử dụng mô hình này ở máy khác:")
    print(f"   model = AutoModelForSequenceClassification.from_pretrained('{repo_id}')")
    print(f"   tokenizer = AutoTokenizer.from_pretrained('{repo_id}')")

if __name__ == "__main__":
    main()
