# Twitter Emotion Recognition System using Fine-Tuned RoBERTa

An end-to-end, production-grade Deep Learning pipeline for Text-Based Emotion Recognition on English Twitter messages. This system fine-tunes the `roberta-base` architecture on the massive `dair-ai/emotion` dataset (~416k samples), incorporating advanced data preprocessing, memory-optimized hyperparameter optimization (Optuna), robust regularizations, and post-processing threshold calibration to overcome severe class imbalance.

## Key Features

- **Advanced Text Preprocessing**: Customized text cleaner engineered for social media text, supporting noise reduction, automated slang restoration (`slang_en.json`), and semantic emoji translation (`emoji_en.json`).
- **Robust Imbalance Management**: Supports dynamic Smoothed Class Weights, Focal Loss ($\gamma=2.0$), and native Label Smoothing to prevent model overconfidence and binary probability saturation.
- **Memory-Optimized Tuning**: Integrated Optuna hyperparameter search with automated data subsampling, gradient accumulation, and aggressive GPU memory cache clearing to safely run on resource-constrained environments (e.g., Google Colab Free).
- **Post-Processing Threshold Calibration**: Replaces standard blind `argmax` decision boundaries with automated Validation-based optimal threshold alignment to balance precision and recall trade-offs for minority classes (`love`, `surprise`).
- **Comprehensive MLOps Evaluation**: Automatically exports publication-ready evaluation artifacts including a high-resolution Confusion Matrix heatmap, localized Error Analysis logs (CSV), structural JSON metrics, and an executive Markdown summary report.
- **Hugging Face Hub Integration**: Dedicated script to seamlessly upload the trained model and custom tokenizers directly to the Hugging Face Cloud Hub securely.

---

## Repository Structure

```text
repository/
|-- configs/                   # Configuration management via YAML files
|   |-- train.yaml             # Hyperparameters for full model training
|   |-- sweep_optuna.yaml      # Search space definition for Optuna tuning
|-- data/                      
|   |-- processed/             # Tokenized and partitioned DatasetDict on disk (saved on Google Drive) 
|   |-- dictionaries/          # External JSON knowledge bases for text cleaning
|       |-- slang_en.json      # English Twitter slang and abbreviation dictionary
|       |-- emoji_en.json      # Emoji-to-text contextual mapping dictionary
|-- results/                   # Evaluation artifacts generated after testing
|   |-- confusion_matrix.png   # Heatmap of the model's test performance
|   |-- error_analysis.csv     # Misclassified samples sorted by model confidence
|   |-- roberta_results.md     # Executive test summary text report
|   |-- roberta_results.json   # Structural metrics log for downstream tracking
|-- scripts/                   # Linear execution pipeline scripts
|   |-- preprocess_data.py     # Cleans, splits (90/5/5), and tokenizes the corpus
|   |-- train.py               # Main model training loop with Early Stopping
|   |-- tune.py                # Hyperparameter optimization sweep via Optuna
|   |-- evaluate.py            # Computes 4-digit metric results and calibration
|   |-- inference.py           # Interactive real-time testing CLI environment
|   |-- push_to_hub.py         # Secures authentication and deploys model to HF Cloud
|-- src/                       # Reusable core modules
|   |-- data_module.py         # PyTorch dataset module & balanced weight calculator
|   |-- model.py               # Custom Trainer, Focal Loss, & evaluation metrics
|   |-- text_cleaner.py        # Custom deterministic text normalization engine
|-- bash_scripts/              # Automated bash execution workflows
|   |-- run_train.sh           
|   |-- run_tune.sh            
|-- requirements.txt           # Project environment dependencies
|-- README.md                  # System documentation
\`\`\`
```

## Installation & Environment Setup
Clone this repository and install the verified environment dependencies:

```Bash
git clone [https://github.com/Kyoyooo/emotion-recognition.git](https://github.com/Kyoyooo/emotion-recognition.git)
cd emotion-recognition
pip install -r requirements.txt
```

## Execution Pipeline (Step-by-Step)
### 1. Data Ingestion & Preprocessing
Downloads the raw unsplit configuration of dair-ai/emotion (416,809 samples), standardizes the raw text via dictionaries, performs a rigorous 90% Train / 5% Validation / 5% Test nested splitting using a fixed deterministic seed, and tokenizes the corpus:

```Bash
python scripts/preprocess_data.py
```
*Outputs are saved directly to ./data/processed.*

### 2. Hyperparameter Sweeping (Optuna)
Runs an automated optimization sweep to discover the ideal combination of learning rate, weight decay, and warmup cycles:

```Bash
./bash_scripts/run_tune.sh
```

### 3. Final Model Training
Trains the model on the full training dataset using the best configuration metrics. It automatically employs early stopping based on Validation F1-score:

```Bash
./bash_scripts/run_train.sh
``` 

*The best model weights are saved at ./saved_models/roberta_emotion_final.*

### 4. Post-Processing Calibration & Test Evaluation
Evaluates the model on the blind Test set. It scans the Validation set first to optimize decision thresholds, calibrates the output probabilities, and exports the full MLOps report suite:

```Bash
python scripts/evaluate.py
```

### 5. Interactive Real-Time Inference
Launches a CLI interface to type custom sentences and visualize predicted emotional states with accurate probability scale distribution under the calibrated thresholds:

```Bash
python scripts/inference.py
```

### 6. Cloud Deployment
Deploys the finalized tokenizers and model weights safely to the Hugging Face Hub:

```Bash
python scripts/push_to_hub.py
```

## Configuration Management
The system parameters are strictly isolated within the ``configs/`` directory.

``configs/train.yaml`` (Example Production Setup)
```YAML
model:
  name: "roberta-base"
  num_labels: 6
  labels_list: ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

data:
  processed_dir: "./data/processed"

training:
  output_dir: "./saved_models/roberta_emotion_weighted"
  final_model_dir: "./saved_models/roberta_emotion_final"
  learning_rate: 2.128e-5        # Optimized via Optuna Trial 1
  weight_decay: 0.0866           # Strong regularization to combat overfitting
  warmup_ratio: 0.1265           # Smooth gradient warmup
  train_batch_size: 16           
  eval_batch_size: 16
  num_epochs: 3                  # Fast convergence on large-scale data
  early_stopping_patience: 1
  fp16: true                     # Mixed-precision training enabled
  loss_type: "weighted_ce"       # Alternatives: "focal_loss"
  smoothing_alpha: 0.5           # Square-root class weights smoothing
  label_smoothing: 0.1           # Prevents probability overconfidence saturation
```

## Experimental Results & Performance Summary
The model yields state-of-the-art results on the blind Twitter evaluation test set ($20,841$ samples):
- Overall Accuracy: 94.40%
- Macro Average F1-score: 92.08%
- Weighted Average F1-score: 94.58%


Detailed Classification Report (4-Decimal Precision)
```Plaintext             
              precision    recall  f1-score   support

     sadness     0.9974    0.9593    0.9780      6038
         joy     0.9985    0.9195    0.9574      7080
        love     0.7732    1.0000    0.8721      1735
       anger     0.9446    0.9520    0.9483      2832
        fear     0.9068    0.9095    0.9082      2365
    surprise     0.7567    0.9987    0.8610       791

    accuracy                         0.9440     20841
   macro avg     0.8962    0.9565    0.9208     20841
weighted avg     0.9525    0.9440    0.9458     20841
``` 
