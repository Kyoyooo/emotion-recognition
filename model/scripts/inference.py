import os
import sys
import yaml
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.text_cleaner import EnglishTextCleaner

def predict_emotion(text, model, tokenizer, cleaner, id2label, device):
    cleaned_text = cleaner.clean(text)
    if not cleaned_text.strip(): return "Unknown", 0.0, ""

    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(**inputs).logits, dim=-1)
        
    confidence, predicted_id = torch.max(probs, dim=-1)
    return id2label[predicted_id.item()], confidence.item(), cleaned_text

def main():
    with open("configs/train.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cleaner = EnglishTextCleaner()
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    model = AutoModelForSequenceClassification.from_pretrained(config['training']['final_model_dir']).to(device).eval()
    
    id2label = {i: label for i, label in enumerate(config['model']['labels_list'])}
    
    print("✅ HỆ THỐNG SẴN SÀNG (Nhập 'q' để thoát)")
    while True:
        user_input = input("\n📝 Text: ")
        if user_input.strip().lower() in ['q', 'quit', 'exit']: break
        if not user_input.strip(): continue
            
        label, conf, cleaned = predict_emotion(user_input, model, tokenizer, cleaner, id2label, device)
        print(f"   [Cleaned]: {cleaned}\n   [Predict]: {label} ({conf*100:.2f}%)")

if __name__ == "__main__":
    main()
