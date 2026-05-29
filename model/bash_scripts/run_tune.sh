#!/bin/bash
export PYTHONPATH=$(pwd)

echo "Khởi chạy quá trình Tối ưu tham số (Optuna Tuning)..."
python scripts/tune.py