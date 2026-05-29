# Đồ án: Nhận diện cảm xúc người dùng mạng xã hội (Emotion Recognition)

## Giới thiệu
Mô hình Transformer (PhoBERT) phân loại cảm xúc văn bản dựa trên tập dữ liệu UIT-VSMEC. Giải quyết 2 bài toán chính: Dữ liệu mạng xã hội nhiễu (Teencode, Emoji) và Mất cân bằng nhãn (Class Imbalance).

## Cài đặt môi trường
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Cách chạy dự án

1. **Tiền xử lý dữ liệu:**
\`\`\`bash
python scripts/preprocess_data.py
\`\`\`

2. **Huấn luyện mô hình:**
\`\`\`bash
./bash_scripts/run_train.sh
\`\`\`

3. **(Tùy chọn) Tối ưu hóa siêu tham số:**
\`\`\`bash
./bash_scripts/run_tune.sh
\`\`\`

4. **Đánh giá mô hình:**
\`\`\`bash
python scripts/evaluate.py
\`\`\`

5. **Chạy suy luận trực tiếp:**
\`\`\`bash
python scripts/inference.py
\`\`\`