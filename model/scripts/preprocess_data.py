import os
import sys
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.text_cleaner import EnglishTextCleaner

def main():
    print("1. Đang khởi tạo Text Cleaner...")
    cleaner = EnglishTextCleaner()
    
    print("2. Đang tải tập dữ liệu dair-ai/emotion (cấu hình 'unsplit' ~416k samples)...")
    # Tải toàn bộ cấu hình unsplit (chỉ chứa một tập train duy nhất)
    raw_dataset = load_dataset("dair-ai/emotion", "unsplit", trust_remote_code=True)["train"]
    
    print("3. Đang phân chia tập dữ liệu (Train 90% / Val 5% / Test 5%)...")
    # Tách 10% ra làm tập Validation và Test
    train_test = raw_dataset.train_test_split(test_size=0.1, seed=42)
    # Chia đôi 10% đó để được 5% Validation và 5% Test
    val_test = train_test['test'].train_test_split(test_size=0.5, seed=42)
    
    # Gộp lại thành cấu trúc DatasetDict chuẩn để các bước sau dùng
    dataset = DatasetDict({
        'train': train_test['train'],
        'validation': val_test['train'],
        'test': val_test['test']
    })
    
    print(f"   - Kích thước tập Train: {len(dataset['train'])} câu")
    print(f"   - Kích thước tập Val:   {len(dataset['validation'])} câu")
    print(f"   - Kích thước tập Test:  {len(dataset['test'])} câu")

    def preprocess_batch(examples):
        cleaned_texts = [cleaner.clean(text) for text in examples["text"]]
        return {"text": cleaned_texts}

    print("4. Đang thực thi tiền xử lý sâu...")
    cleaned_dataset = dataset.map(preprocess_batch, batched=True, num_proc=4)

    print("5. Đang mã hóa dữ liệu với RoBERTa Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def encode_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_dataset = cleaned_dataset.map(
        encode_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"] 
    )

    output_dir = "./data/processed"
    os.makedirs(output_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(output_dir)
    print(f"✅ Hoàn tất! Dữ liệu (Train/Val/Test) đã lưu tại: {output_dir}")

if __name__ == "__main__":
    main()