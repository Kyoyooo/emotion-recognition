import os
from datasets import load_dataset
from transformers import AutoTokenizer
import sys

# Thêm đường dẫn gốc vào sys.path để import được module src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.text_cleaner import VietnameseTextCleaner

def main():
    print("1. Đang khởi tạo bộ làm sạch văn bản (Text Cleaner)...")
    cleaner = VietnameseTextCleaner()
    
    print("2. Đang tải tập dữ liệu tridm/UIT-VSMEC...")
    dataset = load_dataset("tridm/UIT-VSMEC")
    
    # Đổi tên cột cho đồng nhất
    for split in dataset.keys():
        if 'Sentence' in dataset[split].column_names:
            dataset[split] = dataset[split].rename_column("Sentence", "text")
        if 'Emotion' in dataset[split].column_names:
            dataset[split] = dataset[split].rename_column("Emotion", "label")

    # Định nghĩa hàm map cho datasets
    def preprocess_batch(examples):
        # Áp dụng hàm clean cho từng câu trong batch
        cleaned_texts = [cleaner.clean(text) for text in examples["text"]]
        return {"text": cleaned_texts}

    print("3. Đang thực thi tiền xử lý sâu (Deep Preprocessing)...")
    # Áp dụng xử lý đa luồng (num_proc) để tăng tốc độ
    cleaned_dataset = dataset.map(
        preprocess_batch,
        batched=True,
        num_proc=4, # Tùy chỉnh theo số nhân CPU máy bạn
        desc="Làm sạch văn bản"
    )

    # Khởi tạo Tokenizer của PhoBERT để encode dữ liệu
    print("4. Đang mã hóa dữ liệu với PhoBERT Tokenizer...")
    model_name = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Mapping nhãn
    labels_list = ['Sadness', 'Surprise', 'Disgust', 'Fear', 'Anger', 'Other', 'Enjoyment']
    label2id = {label: i for i, label in enumerate(labels_list)}

    def encode_function(examples):
        tokenized = tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )
        tokenized["label"] = [label2id[label] for label in examples["label"]]
        return tokenized

    tokenized_dataset = cleaned_dataset.map(
        encode_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"], # Bỏ cột text vì mô hình chỉ nhận input_ids
        desc="Mã hóa Tokenizer"
    )

    # 5. Lưu dataset đã xử lý xuống thư mục data/processed
    output_dir = "./data/processed"
    os.makedirs(output_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(output_dir)
    print(f"\n✅ Hoàn tất! Dữ liệu đã được tiền xử lý và lưu tại: {output_dir}")
    
    # In thử 1 mẫu để kiểm tra
    print("\n--- SAMPLE KIỂM TRA MẮT THƯỜNG ---")
    sample_raw = dataset['train'][0]
    sample_clean = cleaned_dataset['train'][0]
    print(f"Gốc      : {sample_raw['text']}")
    print(f"Làm sạch : {sample_clean['text']}")

if __name__ == "__main__":
    main()