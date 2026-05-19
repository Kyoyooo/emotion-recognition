# scripts/tune.py
import os
import sys
import yaml
import torch
import optuna
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import WeightedTrainer, compute_metrics
from src.data_module import VSMECDataModule

def load_config(config_path="configs/sweep_optuna.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    
    labels_list = config['model']['labels_list']
    num_labels = config['model']['num_labels']
    id2label = {i: label for i, label in enumerate(labels_list)}
    label2id = {label: i for i, label in enumerate(labels_list)}

    print("1. Đang tải Dữ liệu và tính toán Trọng số qua DataModule...")
    data_module = VSMECDataModule(config['data']['processed_dir'], num_labels)
    train_ds, val_ds, _ = data_module.get_datasets()
    class_weights_tensor = data_module.compute_class_weights()

    def model_init():
        """Optuna gọi hàm này để tạo model mới cho mỗi trial"""
        return AutoModelForSequenceClassification.from_pretrained(
            config['model']['name'],
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )

    def optuna_hp_space(trial):
        """Định nghĩa không gian tìm kiếm (Search Space)"""
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        }

    print("2. Thiết lập TrainingArguments cho các Trial...")
    training_args = TrainingArguments(
        output_dir=config['tuning']['output_dir'],
        per_device_train_batch_size=config['tuning']['train_batch_size'],
        per_device_eval_batch_size=config['tuning']['eval_batch_size'],
        num_train_epochs=config['tuning']['num_epochs'],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=config['tuning']['fp16'],
        report_to="none"
    )

    print("3. Khởi tạo Trainer với cơ chế Hyperparameter Search...")
    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=None, 
        model_init=model_init, 
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config['tuning']['early_stopping_patience'])]
    )

    print(f"\n🚀 BẮT ĐẦU TỐI ƯU HÓA (Chạy {config['tuning']['n_trials']} trials)...")
    best_trial = trainer.hyperparameter_search(
        direction="maximize", 
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=config['tuning']['n_trials']
    )

    print("\n" + "="*50)
    print("🏆 TÌM KIẾM HOÀN TẤT! BỘ THAM SỐ TỐT NHẤT:")
    print("="*50)
    print(f"  Macro F1 đạt được : {best_trial.objective:.4f}")
    print(f"  Learning Rate     : {best_trial.hyperparameters['learning_rate']}")
    print(f"  Weight Decay      : {best_trial.hyperparameters['weight_decay']}")
    print(f"  Warmup Ratio      : {best_trial.hyperparameters['warmup_ratio']}")
    print("="*50)

if __name__ == "__main__":
    main()