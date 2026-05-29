import os
import sys
import yaml
from transformers import AutoModelForSequenceClassification, TrainingArguments, EarlyStoppingCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import WeightedTrainer, compute_metrics
from src.data_module import EmotionDataModule

def main():
    with open("configs/train.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    
    labels_list = config['model']['labels_list']
    num_labels = config['model']['num_labels']
    id2label = {i: label for i, label in enumerate(labels_list)}
    label2id = {label: i for i, label in enumerate(labels_list)}

    data_module = EmotionDataModule(config['data']['processed_dir'], num_labels)
    train_ds, val_ds, _ = data_module.get_datasets()
    class_weights_tensor = data_module.compute_class_weights()

    model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['name'], num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=config['training']['weight_decay'],
        warmup_ratio=config['training']['warmup_ratio'],
        per_device_train_batch_size=config['training']['train_batch_size'],
        per_device_eval_batch_size=config['training']['eval_batch_size'],
        num_train_epochs=config['training']['num_epochs'],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=config['training']['fp16'],
        lr_scheduler_type="cosine", # Tối ưu chống overfitting bằng Cosine Decay
        report_to="none"
    )

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        loss_type=config['training'].get('loss_type', 'weighted_ce'),
        gamma=config['training'].get('focal_loss_gamma', 2.0),
        label_smoothing=config['training'].get('label_smoothing', 0.0), # Truyền tham số vào đây
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config['training']['early_stopping_patience'])]
    )

    trainer.train()
    trainer.save_model(config['training']['final_model_dir'])
    print(f"✅ Đã lưu model tại: {config['training']['final_model_dir']}")

if __name__ == "__main__":
    main()
