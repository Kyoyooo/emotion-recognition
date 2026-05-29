import os
import sys
import yaml
import optuna
import gc          # Thư viện dọn dẹp bộ nhớ (Garbage Collector)
import torch       # Thư viện kiểm soát GPU
from transformers import AutoModelForSequenceClassification, TrainingArguments, EarlyStoppingCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import WeightedTrainer, compute_metrics
from src.data_module import EmotionDataModule

def main():
    with open("configs/sweep_optuna.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    
    num_labels = config['model']['num_labels']
    labels_list = config['model']['labels_list']
    id2label = {i: label for i, label in enumerate(labels_list)}
    label2id = {label: i for i, label in enumerate(labels_list)}

    print("1. Đang tải Dữ liệu...")
    data_module = EmotionDataModule(config['data']['processed_dir'], num_labels)
    train_ds, val_ds, _ = data_module.get_datasets()
    
    # Lấy trọng số nhãn dựa trên tập data gốc
    class_weights_tensor = data_module.compute_class_weights()

    # ------------------------------------------------------------------
    # TỐI ƯU 1: DATA SUBSAMPLING (TRÍCH MẪU TẬP DỮ LIỆU)
    # Không ai dùng toàn bộ 400k mẫu chỉ để tìm siêu tham số cả.
    # Chúng ta sẽ lấy ngẫu nhiên 10% dữ liệu Train và 30% Val để Tuning.
    # Việc này giúp giải phóng hàng GB RAM hệ thống và tăng tốc x10 lần.
    # ------------------------------------------------------------------
    print("2. Đang trích xuất tập con (Subsampling) để chống tràn RAM...")
    subset_train_size = int(len(train_ds) * 0.1) # Lấy 10%
    subset_val_size = int(len(val_ds) * 0.3)     # Lấy 30%
    
    train_ds = train_ds.shuffle(seed=42).select(range(subset_train_size))
    val_ds = val_ds.shuffle(seed=42).select(range(subset_val_size))
    print(f"   -> Dùng {subset_train_size} mẫu Train và {subset_val_size} mẫu Val cho Tuning.")

    def model_init():
        # ------------------------------------------------------------------
        # TỐI ƯU 2: AGGRESSIVE GARBAGE COLLECTION (DỌN RÁC BỘ NHỚ)
        # Mỗi khi Optuna bắt đầu một trial mới, ta ép Python và PyTorch 
        # phải xóa sạch model cũ khỏi bộ nhớ GPU (VRAM).
        # ------------------------------------------------------------------
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return AutoModelForSequenceClassification.from_pretrained(
            config['model']['name'], num_labels=num_labels, id2label=id2label, label2id=label2id
        )

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        }

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

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=None, model_init=model_init, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config['tuning']['early_stopping_patience'])]
    )

    best_trial = trainer.hyperparameter_search(
        direction="maximize", backend="optuna", hp_space=optuna_hp_space, n_trials=config['tuning']['n_trials']
    )
    print(f"🏆 Best F1: {best_trial.objective:.4f}")

if __name__ == "__main__":
    main()
