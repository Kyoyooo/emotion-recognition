# Twitter Emotion Recognition System using Transformer-based NLP model (RoBERTa)

An end-to-end, production-grade Deep Learning pipeline for Text-Based Emotion Recognition on English Twitter messages. This system fine-tunes the `roberta-base` architecture on the massive **[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)** dataset (~416k samples), incorporating advanced data preprocessing, memory-optimized hyperparameter optimization (Optuna), robust regularizations, and post-processing threshold calibration to overcome severe class imbalance.

## Key Features

- **Advanced Text Preprocessing**: Customized text cleaner engineered for social media text, supporting noise reduction, automated slang restoration (`slang_en.json`), and semantic emoji translation (`emoji_en.json`).
- **Robust Imbalance Management**: Supports dynamic Smoothed Class Weights, Focal Loss ($\gamma=2.0$), and native Label Smoothing to prevent model overconfidence and binary probability saturation.
- **Memory-Optimized Tuning**: Integrated Optuna hyperparameter search with automated data subsampling, gradient accumulation, and aggressive GPU memory cache clearing to safely run on resource-constrained environments (e.g., Google Colab Free).
- **Post-Processing Threshold Calibration**: Replaces standard blind `argmax` decision boundaries with automated Validation-based optimal threshold alignment to balance precision and recall trade-offs for minority classes (`love`, `surprise`).
- **Comprehensive MLOps Evaluation**: Automatically exports publication-ready evaluation artifacts including a high-resolution Confusion Matrix heatmap, localized Error Analysis logs (CSV), structural JSON metrics, and an executive Markdown summary report.
- **Hugging Face Hub Integration**: Dedicated script to seamlessly upload the trained model and custom tokenizers directly to the Hugging Face Cloud Hub securely.

## Dataset Specifications & Large Data Storage
The model is trained on the comprehensive **[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)** containing a total of 416,809 records under its flat unsplit configuration.

### 1. Data Split Architecture
To prevent any risk of data leakage, a nested splitting algorithm partitions the text into three strict, deterministic splits utilizing a fixed evaluation seed:
- Training Set (90%): $375,128$ samples — used for loss optimization and backpropagation.
- Validation Set (5%): $20,840$ samples — used for hyperparameter evaluation, early stopping, and boundary calibration.
- Test Set (5%): $20,841$ samples — a completely blind partition used solely for generalized model reporting.

### 2. Google Drive Caching for Large Vector Arrays
Because the fully preprocessed and tokenized tensor structures inside ``model/data/processed/`` exceed Git file system capacities, they are hosted outside repository memory.
- 📁 Google Drive Active Cache Directory: Download the processed splits from this [Google Drive Directory Archive](https://drive.google.com/drive/folders/14PCN39rOlkOP3gAzj_0FXcYWQjPgkvFi).
- Setup: Extract the file content blocks directly into the ``model/data/processed/`` path prior to running training operations.
---

## Repository Structure

```text
repository/
|-- model/                         # Model Layer 
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
|-- emotion-recognition-app/       # Application Layer 
```

## Deep Learning Engineering Layer (``model/``)
### 1. Deep Text Preprocessing Pipeline
Every sentence from the raw data stream is routed through a 5-stage cleaning pipeline implemented inside ``model/src/text_cleaner.py``:
- **Structural Noise Reduction**: Lowercases all incoming texts and applies regular expressions (Regex) to strip off text anomalies such as URL links, structural HTML tags, user ``@mentions``, and hashtags (``#``).
- **Contextual Emoji Translation**: Uses the ``emoji_en.json`` asset map to identify and convert emoticons into equivalent plain English emotional keywords (e.g., 🥰  → *passionate/adorable*), preserving vital semantic details.
- **Slang & Abbreviation Mapping**: Translates informal internet jargon (e.g., *im  → i am, dont  → do not, u  → you*) via ``slang_en.json`` to restore sentences to standardized grammar before tokens hit the transformer layer.
- **Repeated Character Contraction**: Truncates emotional character exaggerations (e.g., *loooove  → love*, *happyyyyy  → happy*) using string patterns to eliminate out-of-vocabulary anomalies.
- **Byte-Pair Encoding (BPE) Tokenization**: Encodes clean strings using RoBERTa's native 50,265 token vocabulary. Special tokens (``<s>`` and ``</s>``) enclose the string, and a rigid boundaries constraint pads or truncates sequences to a uniform vector length of ``max_length = 128``.

### 2. Hyperparameter Sweeping with Optuna
To establish highly optimized learning constraints, ``model/scripts/tune.py`` executes hyperparameter optimization loops over learning rates, weight decays, and warmup cycles.

To circumvent Out-Of-Memory (OOM) failures under hardware resource limits (e.g., Google Colab Free), the tuning framework implements memory-efficient mechanisms:
- **Subsampling Optimization**: Isolates an informative representative subset ($10\%$ of Train, $30\%$ of Validation) to expedite trials.
- **Gradient Accumulation**: Uses a physical ``batch_size = 8`` combined with ``gradient_accumulation_steps = 4`` to accurately simulate a large batch size of 32 while reducing VRAM footprints.
- **Memory Management**: Interleaves PyTorch's cache eviction (``torch.cuda.empty_cache()``) and garbage collector sweeps (``gc.collect()``) inside the ``model_init`` routine.

The search objective optimizes a composite function factoring in both overall stability and rare class performance:
$$\text{Objective Value} = \text{Accuracy} + \text{Macro } F_1\text{-score}$$

Optuna finalized 5 separate execution sweeps (Trials 0 to 4), locking down peak performance at Trial 1 (Objective Score: 1.8482):
- ``learning_rate``: $2.1286 \times 10^{-5}$
- ``weight_decay``: $0.0866$ (High regularization bounding protects the Transformer weights from memorizing noisy Twitter slang)
- ``warmup_ratio``: $0.1265$

### 3. Model Training Lifecycle
The execution of ``model/scripts/train.py`` trains the ``roberta-base`` classifier across the full $375,128$ dataset rows for 3 epochs utilizing the optimized Trial 1 hyperparameters. It incorporates a Cosine Learning Rate Scheduler and an ``EarlyStoppingCallback`` with a ``patience=1`` threshold constraint against the Validation Macro $F_1$ score to intercept overfitting immediately at the point of saturation.

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

Analytical InsightsThe Semantic Ceiling: 
1. An overall accuracy of 94.40% is highly competitive, approaching the limit of human inter-annotator agreement on brief social media text.
2. Precision vs. Recall Control: Minority classes such as love and surprise exhibit near-perfect recall ($\ge 99.8\%$), capturing almost every true positive instance. The slight drop in precision is caused by overlapping semantic context boundaries inherent to human emotion data (e.g., highly energetic joy phrases like "passionate about coding" being predicted as love, or extreme fear phrases like "completely overwhelmed" overlapping with surprise).

## Application Layer Local Setup (``emotion-recognition-app/``)
### 1. Launch Shared Storage Infrastructure
Initialize the docker containers holding background microservices (PostgreSQL for transaction memory and Redis for BullMQ handling):

```Bash
docker compose up -d
```

### 2. Boot the Python FastAPI Inference Service
```Bash
cd emotion-recognition-app/apps/model-api
python -m venv .venv

# Windows OS activation
.venv\\Scripts\\activate
# macOS / Linux OS activation
source .venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Spin up the Node.js API Gateway Core
```Bash
cd emotion-recognition-app/apps/api
npm install
npm run prisma:generate
npm run prisma:migrate
npm run dev
```

### 4. Deploy the Next.js Client User Interface
```Bash
cd emotion-recognition-app/apps/web
npm install
npm run dev
``` 
*Note: To initialize all Node.js package workspaces simultaneously from the monorepo root layer, simply use: ``npm install``.*

Open and monitor the analytics frontend dashboard via: http://localhost:3000.

## System Verification & Testing
Verify that all background endpoints are running correctly using the built-in health routes:
- **Model Inference Engine Route**: http://localhost:8000/health
- **API Node Gateway Route**: http://localhost:4000/health
  
To validate production readiness, check static types, and compile code packages, run:

```Bash
npm run lint
npm run typecheck
npm run build
npm run test
```

## Production Deployment Guide
- **Frontend Dashboard UI**: Host directly on Vercel with the root build parameter locked into ``emotion-recognition-app/apps/web``. Ensure ``NEXT_PUBLIC_API_URL`` routes to your remote API Gateway.
- **Express API Gateway Backend**: Deploy onto Render, Railway, or Fly.io. Provide production environmental tags for ``DATABASE_URL``, ``REDIS_URL``, ``CORS_ORIGIN``, and ``MODEL_API_URL``.
- **Model Inference API Service**: Deploy onto dedicated GPU instances or cloud target environments like Hugging Face Spaces. Extend cold start timeout parameters to account for model weights download and initialization.

## Project Contributors
This monorepo was engineered as a final group deliverable for the **Statistical Learning course (CSC15004)** at VNU-HCM University of Science, Faculty of Information Technology:
- **Võ Trần Duy Hoàng** - Student ID: ``23120266``
- **Trương Sỹ Khánh** - Student ID: ``23120284``
- **Lê Công Phúc** - Student ID: ``23120330``

**Academic Supervisors**: Ngô Minh Nhựt, Lê Long Quốc.
**Official Project Submission Date**: May 30, 2026.
