#!/bin/bash
# Đặt thư mục gốc của project vào PYTHONPATH
export PYTHONPATH=$(pwd)

echo "Khởi chạy quá trình Huấn luyện (Training)..."
python scripts/train.py