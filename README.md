# Social Media Emotion Recognition System (Monorepo)
An end-to-end, production-oriented monorepo implementing a high-performance system for Text-Based Emotion Recognition on informal English text. The system features a core Deep Learning pipeline built on a fine-tuned roberta-base transformer architecture, coupled with a robust 3-tier web application architecture designed to ingest noisy social media data, track model performance, and serve real-time or async batch predictions at scale.

## System Architecture
The project is structured as a unified monorepo divided into two foundational layers: the Deep Learning Engineering Layer (model training, tuning, and local verification) and the Application Infrastructure Layer (web deployment, async workers, and databases).


```Plaintext

[Client Layer] Next.js Frontend (apps/web)
                      │
                      ▼ (REST / Batch Requests)
[Backend Layer] Node.js Express API Gateway (apps/api) ◄──► Redis / BullMQ Task Queue
                      │                                            │
                      ▼ (Internal Microservice Routing)            ▼
[Inference Layer] FastAPI Model Registry (apps/model-api) ◄──── Async Workers
                      │
                      ▼
  PostgreSQL Database (Prisma ORM Persistence)
```

### 1. Core Model Pipeline (Deep Learning Layer)
- **Pre-trained Foundation**: Robustly Optimized BERT Approach (roberta-base) customized to map multi-class emotional profiles on social media semantics
- **Imbalance & Optimization Techniques**: Implements dynamic Smoothed Class Weights, Focal Loss, and Cross-Entropy with Label Smoothing to counteract heavy class imbalances and prevent model overconfidence.  
- **Post-Processing Calibration**: Evaluates validation-set outputs to compute a custom optimal threshold map. Probabilities are scaled post-inference to dramatically boost minority class precision without sacrificing recall.  

### 2. Microservices & Web Infrastructure (Application Layer)
- **Online Execution**: Real-time evaluation flows sequentially:

$$\text{Next.js App} \rightarrow \text{Express API Gateway} \rightarrow \text{FastAPI Model Registry} \rightarrow \text{PostgreSQL}$$

- **Offline Batch Execution**: High-volume asynchronous text processing relies on a distributed task architecture:

$$\text{Next.js File Upload} \rightarrow \text{Express API} \rightarrow \text{Redis / BullMQ Queue} \rightarrow \text{Background Worker} \rightarrow \text{FastAPI Inference} \rightarrow \text{PostgreSQL Storage}$$

- **Design Philosophy**: Adheres to a pragmatic route-controller-service structure optimized specifically for rapid validation and microservice vertical isolation.

## Project Structure
```Plaintext
repository/
|-- apps/                          # Application Infrastructure Layer (Web Apps)
|   |-- api/                       # Gateway Node.js Express API (TypeScript, Prisma)
|   |-- model-api/                 # Microservice FastAPI Inference Service (PyTorch)
|   |-- web/                       # Client Next.js Frontend Dashboard (Tailwind, Recharts)
|-- configs/                       # Configuration Management Directory
|   |-- train.yaml                 # Parameters for production model training
|   |-- sweep_optuna.yaml          # Search space limits for Optuna tuning sweeps
|-- data/                          # Data Management Directory
|   |-- processed/                 # Tokenized structural DatasetDict (Hosted on Google Drive)
|   |-- dictionaries/              # Text normalization assets
|       |-- slang_en.json          # Dictionary for Twitter abbreviation restoration
|       |-- emoji_en.json          # Dictionary for textual emoji translation
|-- results/                       # Local execution output evaluation artifacts
|   |-- confusion_matrix.png       # High-resolution performance evaluation heatmap
|   |-- error_analysis.csv         # Fault analysis tracker sorted by model confidence
|   |-- roberta_results.md         # Formatted text report summarizing test stats
|   |-- roberta_results.json       # Structural metrics cache for deployment tracking
|-- scripts/                       # Deep Learning Lifecycles (Python Execution Scripts)
|   |-- preprocess_data.py         # Ingests, normalizes, splits, and compiles the dataset
|   |-- train.py                   # Main supervised training loop with Early Stopping
|   |-- tune.py                    # Hyperparameter grid search using Optuna Subsampling
|   |-- evaluate.py                # Computes 4-decimal evaluation reports and artifact dumps
|   |-- inference.py               # Interactive CLI terminal for localized real-time tests
|   |-- push_to_hub.py             # Securely authenticates and deploys artifacts to HF Hub
|-- src/                           # Shared Deep Learning pipeline source modules
|   |-- data_module.py             # Prebuilt Dataset loader and class balance calculator
|   |-- model.py                   # Custom CustomLossTrainer interface and Focal Loss core
|   |-- text_cleaner.py            # Social media cleaning regex tokenizer engine
|-- bash_scripts/                  # Standard execution shells
|   |-- run_train.sh               
|   |-- run_tune.sh                
|-- docker-compose.yml             # External infrastructure orchestrator (PostgreSQL & Redis)
|-- requirements.txt               # Global Python library dependency stack
|-- package.json                   # Root configuration managing global npm workspaces
```

## Dataset Specifications & Data Split Map
The core system executes deep learning iterations on the public **[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)** Twitter corpus.  

### 1. Volume & Partition Layout
The dataset uses the flat ``unsplit`` configuration containing a total of 416,809 samples. The preprocessing script executes a nested random-seeded partitioning loop to split the data into 3 deterministic segments:  
- Train Set (90%): $375,128$ samples. Used for updating network weights via backpropagation.  
- Validation Set (5%): $20,840$ samples. Used for early stopping checks, regularizations, and boundary calibration.  
- Test Set (5%): $20,841$ samples. Kept completely blind to evaluate final generalizability.

### 2. Large Data Handling Note
Because the tokenized and vectorized representations inside the ``data/processed/`` folder exceed standard version-control storage thresholds, the compiled cache has been exported externally.
 Data Hosting Link: Download the compiled structural artifact from this [Google Drive Directory Archive](https://drive.google.com/drive/folders/14PCN39rOlkOP3gAzj_0FXcYWQjPgkvFi). Extract its internal folds directly into your local ``data/processed/`` path prior to running training scripts.

## Local Development Setup
### 1. Start External Infrastructure
Spin up the local containerized instances for PostgreSQL (data store) and Redis (BullMQ task engine):

```Bash
docker compose up -d
```

### 2. Initialize and Run the Model Inference API
```Bash
cd apps/model-api
python -m venv .venv

# Windows activation
.venv\Scripts\activate
# Linux / macOS activation
source .venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Initialize and Run the Backend API Gateway
From the root workspace or target directory:

```Bash
cd apps/api
npm install
npm run prisma:generate
npm run prisma:migrate
npm run dev
```

### 4. Initialize and Run the Frontend Client Application
```Bash
cd apps/web
npm install
npm run dev
```

*Alternatively, install all Node.js workspace dependencies concurrently from the monorepo root via:*

```Bash
npm install
```
The client UI dashboard will be accessible at http://localhost:3000.

## Experimental Metrics & Verification
Following hyperparameter tuning using Optuna, the optimal configuration parameters were locked inside ``configs/train.yaml``:
- ``learning_rate``: $2.1286 \times 10^{-5}$   
- ``weight_decay``: $0.0866$   
- ``warmup_ratio``: $0.1265$   

Test Performance Report (4-Decimal Precision)
When validated against the blind test partition ($20,841$ verification elements) , the model achieves excellent classification metrics:  

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

All calculated evaluation metrics, metadata parameters, classification strings, and matrix dimensions are exported to ``results/roberta_results.json`` and ``results/roberta_results.md`` automatically upon completing script evaluation executions.

## Health Checks & Verification
Ensure your microservices are live and operating correctly by checking their endpoints:
- **Model Inference Engine**: http://localhost:8000/health
- **Backend API Gateway**: http://localhost:4000/health

To validate system types and build performance stability across all JavaScript/TypeScript modules, run:

```Bash
npm run lint
npm run typecheck
npm run build
npm run test
``` 

## Production Deployment Guide
- **Frontend Client UI**: Deploy to Vercel with the root context set to ``apps/web``. Ensure ``NEXT_PUBLIC_API_URL`` points to your public Node.js Express server.

- **Backend API Gateway**: Deploy to platforms like Render, Railway, or Fly.io. Configure your environment variables for ``DATABASE_URL``, ``REDIS_URL``, ``CORS_ORIGIN``, and ``MODEL_API_URL``.

- **Model Inference API**: Host on an instance with GPU availability or specialized container spaces (e.g., Hugging Face Spaces). Set your cold-start timeouts to accommodate downloading the pre-trained weights during initialization.

## Contributors & Team
This monorepo was developed as part of a final project for the **Statistical Learning course (CSC15004)** at **VNU-HCM University of Science, Faculty of Information Technology**:  
- **Võ Trần Duy Hoàng** - Student ID: ``23120266`` 
- **Trương Sỹ Khánh** - Student ID: ``23120284``   
- **Lê Công Phúc** - Student ID: ``23120330``  

**Academic Instructors**: Ngô Minh Nhựt , Lê Long Quốc.
**Submission Date**: May 30, 2026