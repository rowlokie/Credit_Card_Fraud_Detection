# 🔍 Credit Card Fraud Detection

> A **production-grade** machine learning system for detecting fraudulent credit card transactions. Features Optuna hyperparameter optimization, SHAP explainability, FastAPI REST backend, Streamlit dashboard, MLflow experiment tracking, Docker deployment, and CI/CD.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple?style=for-the-badge)

---

## ⭐ Features

| Feature | Description |
|---------|-------------|
| **Cross-Validation + Optuna** | Bayesian hyperparameter optimization with Stratified K-Fold CV |
| **SHAP Explainability** | TreeExplainer with summary, waterfall, and per-prediction explanations |
| **FastAPI + Streamlit** | Decoupled REST API backend + interactive frontend dashboard |
| **Batch Prediction** | CSV upload with batch processing via API or Streamlit |
| **Threshold Optimization** | F1, Youden's J, cost-sensitive, and PR break-even strategies |
| **MLflow Tracking** | Experiment tracking with params, metrics, models, and artifacts |
| **Docker** | Multi-stage build with Docker Compose (API + Frontend + MLflow) |
| **Logging & Config** | Structured logging + YAML config + environment variable overrides |
| **Unit Tests** | Comprehensive pytest suite with fixtures and coverage |
| **GitHub Actions CI/CD** | Lint, test, build, and security scanning pipeline |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Compose                           │
│                                                              │
│  ┌──────────────┐    HTTP     ┌───────────────────┐         │
│  │  Streamlit    │ ─────────→ │   FastAPI          │         │
│  │  Frontend     │            │   Backend          │         │
│  │  (port 8501)  │ ←───────── │   (port 8000)      │         │
│  └──────────────┘             │                     │         │
│                                │  POST /predict      │         │
│  ┌──────────────┐             │  POST /predict/batch │        │
│  │   MLflow      │             │  POST /predict/explain│       │
│  │  Tracking     │             │  GET  /health        │        │
│  │  (port 5000)  │             │  GET  /model/info    │        │
│  └──────────────┘             └───────────────────┘         │
│                                        │                     │
│                           ┌────────────┴────────────┐       │
│                           │   ML Pipeline Core      │       │
│                           │  • Optuna Optimizer      │       │
│                           │  • SHAP Explainer        │       │
│                           │  • Threshold Optimizer   │       │
│                           │  • MLflow Logger         │       │
│                           └─────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-repo/creditfrauddetection.git
cd creditfrauddetection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

### 2. Download Dataset

Download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in `data/`.

### 3. Train Models

```bash
# Standard training
python train.py

# With hyperparameter optimization (Optuna)
python train.py --optimize --n-trials 50

# Skip optional steps
python train.py --no-eda --no-shap
```

### 4. Run the Application

```bash
# Start FastAPI backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Start Streamlit frontend (in another terminal)
streamlit run app.py

# Or use Docker Compose
docker-compose up -d
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Single transaction prediction |
| `POST` | `/predict/batch` | Batch CSV upload prediction |
| `POST` | `/predict/explain` | Prediction with SHAP explanation |
| `GET` | `/health` | API health check |
| `GET` | `/model/info` | Model information and metrics |
| `GET` | `/model/threshold` | Current optimal threshold |
| `PUT` | `/model/reload` | Hot-reload model artifacts |

**Interactive docs:** http://localhost:8000/docs

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

# Skip slow/integration tests
pytest tests/ -m "not slow and not integration"
```

---

## 🐳 Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# Services:
#   API:      http://localhost:8000
#   Frontend: http://localhost:8501
#   MLflow:   http://localhost:5000

# Stop
docker-compose down
```

---

## 📊 MLflow Experiment Tracking

```bash
# Start MLflow UI
mlflow ui --port 5000

# View at http://localhost:5000
```

---

## 📁 Project Structure

```
creditfrauddetection/
├── api/                          # FastAPI backend
│   ├── main.py                   # API endpoints
│   ├── schemas.py                # Pydantic models
│   └── dependencies.py           # DI & model loading
├── src/                          # Core ML pipeline
│   ├── config.py                 # Configuration (YAML + env vars)
│   ├── logging_config.py         # Structured logging
│   ├── data_loader.py            # Dataset loading & validation
│   ├── preprocessing.py          # Scaling, PCA, splitting
│   ├── balancing.py              # SMOTE class balancing
│   ├── models.py                 # Model creation & serialization
│   ├── evaluation.py             # Metrics & comparison
│   ├── optimizer.py              # Optuna + RandomizedSearchCV
│   ├── threshold.py              # Threshold optimization
│   ├── explainability.py         # SHAP integration
│   ├── tracking.py               # MLflow wrapper
│   └── visualizations.py         # Plot generation
├── tests/                        # Pytest suite
├── .github/workflows/            # CI/CD pipelines
│   ├── ci.yml                    # Lint + Test + Build
│   └── train.yml                 # Model retraining
├── app.py                        # Streamlit frontend
├── train.py                      # Training pipeline
├── predict.py                    # CLI predictions
├── config.yaml                   # Configuration file
├── Dockerfile                    # Multi-stage Docker build
├── docker-compose.yml            # Multi-service orchestration
├── requirements.txt              # Production dependencies
└── requirements-dev.txt          # Development dependencies
```

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | 99.96% | 95.74% | 76.53% | 0.8511 | 0.9780 |
| **XGBoost** | 99.96% | 89.32% | 82.65% | 0.8586 | 0.9826 |

*Results may improve with `--optimize` flag (Optuna hyperparameter tuning).*

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

Built with ❤️ by Piyush Batavale
