# Customer Churn Prediction — MLOps Project

**End-to-end ML service that predicts telecom customer churn with production-grade deployment, monitoring, and reproducibility.**

Built for a Semester 8 undergraduate MLOps course. Emphasizes reproducibility, containerization, cloud deployment, and observability — not just model accuracy.

**Key metrics:**
- **Model:** Random Forest (rf-v1) with ROC-AUC = 0.8434 (test set)
- **Dataset:** IBM Telco Customer Churn — 7,043 customers, 21 features, 26.5% churn rate
- **Production:** Deployed on Render.com at https://churn-api-latest-lbqq.onrender.com
- **Pipeline:** Data validation → preprocessing → training (6 models) → MLflow → FastAPI → Docker → Render → Prometheus + Grafana monitoring

---

## Project Structure

<<<<<<< HEAD
```
├── churn_mlops.ipynb                    # Full training pipeline (data → model → artifacts)
├── app/
│   ├── main.py                          # FastAPI service with /predict, /health, /metrics
│   └── requirements.txt                 # Python dependencies
├── tests/
│   ├── test_api.py                      # API integration tests
│   └── test_preprocess.py               # Preprocessing pipeline tests
├── artifacts/
│   ├── model.pkl                        # Random Forest model (rf-v1)
│   ├── preprocessor.pkl                 # ColumnTransformer (StandardScaler + OneHotEncoder)
│   └── schema.json                      # Input schema + metadata
├── experiments/
│   └── experiment_log.csv               # 6 model comparison (ROC-AUC, F1, Recall, etc.)
├── monitoring/
│   ├── prometheus.yml                   # Scrape config (local API + live Render endpoint)
│   ├── prometheus_datasource.yml        # Grafana datasource provisioning
│   ├── dashboard_provider.yml           # Grafana dashboard provider config
│   └── grafana_dashboard.json           # 9-panel dashboard (latency, throughput, predictions, CPU, memory)
├── docker-compose.yml                   # Local stack: API (8001) + Prometheus (9090) + Grafana (3000)
├── Dockerfile                           # Python 3.11-slim, libgomp1 for LightGBM compat
├── .gitignore                           # Excludes mlruns/, artifacts/*, .env, etc.
├── README.md                            # This file
└── WA_Fn-UseC_-Telco-Customer-Churn.csv # Input dataset (7,043 rows)
=======
```bash
git clone https://github.com/mohameA337/churn-predection-mlops-project.git
cd churn-mlops
pip install -r api/requirements.txt
>>>>>>> 8a77a60483c24d0127692ba2dd4609dbd91572e8
```

---

## Setup & Installation

**Prerequisites:** Git, Python 3.11+, Docker + Docker Compose.

```bash
# Clone repository
git clone https://github.com/mohameA337/churn-predection-mlops-project.git
cd churn-predection-mlops-project

# Install dependencies
pip install -r app/requirements.txt
pip install jupyter pandas scikit-learn xgboost lightgbm imbalanced-learn mlflow pytest
```

**Dataset:**
- Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from IBM's public Telco dataset
- Place in project root
- Notebook validates schema automatically on load

---

## Run Training Pipeline

Open and run all cells in `churn_mlops.ipynb`.

- Artifacts are saved to `artifacts/` (model.pkl, preprocessor.pkl, schema.json)
- Experiment log saved to `experiments/experiment_log.csv`
- MLflow runs stored in `mlruns/`

**View MLflow experiment UI:**
```bash
mlflow ui
# Open: http://localhost:5000
```

---

## Run Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

---

## Build & Run Docker (local)

```bash
# Build image
docker build -t churn-api .

# Run API only (port 8000)
docker run -d -p 8000:8000 churn-api

# Run full stack: API + Prometheus + Grafana
docker-compose up -d
# API:        http://localhost:8001
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin / admin)
```

---

## API Usage

### Health check
```bash
curl http://localhost:8001/health
```

**Response:**
```json
{"status": "ok", "model_version": "rf-v1"}
```

### Predict churn
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 2,
    "MonthlyCharges": 89.5,
    "TotalCharges": 179.0,
    "SeniorCitizen": 0,
    "gender": "Female",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
  }'
```

**Response:**
```json
{
  "churn_probability": 0.9521,
  "churn_prediction": true,
  "model_version": "rf-v1"
}
```

### Interactive API docs
Open `http://localhost:8001/docs` in a browser.

---

## Monitoring

After `docker-compose up -d`:

| Service | URL | Credentials |
|---|---|---|
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |

The Prometheus datasource and Grafana dashboard are provisioned automatically on first start.

Key metrics tracked:
- Request latency (p50, p95, p99)
- Throughput (req/sec)
- Error rate
- CPU and memory usage
- Churn prediction distribution (True vs False)
- Churn probability histogram (model-level view)

---

## Cloud Deployment (Render.com)

The API is deployed publicly at: **https://churn-api-latest-lbqq.onrender.com**

Test with:
```powershell
# Health check
curl.exe https://churn-api-latest-lbqq.onrender.com/health

# Prediction
$body = '{"tenure":2,"MonthlyCharges":89.5,"TotalCharges":179.0,"SeniorCitizen":0,"gender":"Female","Partner":"No","Dependents":"No","PhoneService":"Yes","MultipleLines":"Yes","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check"}'
Invoke-RestMethod -Uri https://churn-api-latest-lbqq.onrender.com/predict -Method POST -ContentType "application/json" -Body $body
```

**Monitoring live Render metrics:** Prometheus scrapes the Render endpoint every 15 seconds. View metrics in Grafana (local) at `http://localhost:3000` after `docker-compose up -d`.

---

## Experiment Results

| Model | Val ROC-AUC | Val F1 | Val Recall | Val Precision | Val Accuracy |
|---|---|---|---|---|---|
| **Random Forest** | **0.8344** | **0.6368** | 0.7082 | 0.5785 | 0.7852 |
| RF + SMOTE | 0.8333 | 0.6295 | 0.6833 | 0.5836 | 0.7862 |
| Logistic Regression | 0.8291 | 0.6136 | 0.7687 | 0.5143 | 0.7427 |
| XGBoost | 0.8249 | 0.6021 | 0.7189 | 0.5179 | 0.7474 |
| LightGBM | 0.8179 | 0.6109 | 0.7011 | 0.5412 | 0.7625 |
| Decision Tree | 0.8089 | 0.5857 | 0.7722 | 0.4717 | 0.7096 |

### Final Model: Random Forest (rf-v1)

**Selection logic:** Automatic selection by highest validation ROC-AUC among [Random Forest, RF+SMOTE].

**Test Set Performance:**
- Accuracy: **0.7947**
- Precision: **0.5785**
- Recall: **0.7082**
- F1-score: **0.6368**
- ROC-AUC: **0.8434** ✓

**Why ROC-AUC?** It measures how well the model ranks churners vs non-churners across all decision thresholds, independent of class imbalance. At 0.8434, the model achieves industry-standard performance (benchmark: 0.80+).

**Class imbalance handling:**
- Dataset: 73.5% No churn / 26.5% Churn (7,043 customers)
- Applied `class_weight='balanced'` in all models
- SMOTE (Synthetic Minority Over-sampling) comparison: RF without SMOTE outperformed RF+SMOTE on ROC-AUC, F1, and Recall — confirmed that balanced weighting was sufficient

Full details: `experiments/experiment_log.csv` or `mlflow ui` → http://localhost:5000

---

## Dataset Schema

| Column | Type | Description |
|---|---|---|
| tenure | int | Months with company (0–72) |
| MonthlyCharges | float | Monthly bill in USD |
| TotalCharges | float | Total amount billed |
| SeniorCitizen | int | 1 = senior citizen |
| gender | str | Male / Female |
| Partner | str | Yes / No |
| Dependents | str | Yes / No |
| PhoneService | str | Yes / No |
| MultipleLines | str | Yes / No / No phone service |
| InternetService | str | DSL / Fiber optic / No |
| OnlineSecurity | str | Yes / No / No internet service |
| Contract | str | Month-to-month / One year / Two year |
| PaperlessBilling | str | Yes / No |
| PaymentMethod | str | Electronic check / Mailed check / Bank transfer / Credit card |
| **Churn** | int | **Target** — 1 = churned, 0 = stayed |

Train / Val / Test split: **70% / 15% / 15%**, stratified on Churn.

---

## Quick Links

| Resource | URL |
|---|---|
| **Live API** | https://churn-api-latest-lbqq.onrender.com |
| **API Docs** | https://churn-api-latest-lbqq.onrender.com/docs |
| **Local Grafana** | http://localhost:3000 (admin / admin) |
| **Local Prometheus** | http://localhost:9090 |
| **MLflow UI** | http://localhost:5000 |
| **Repo** | https://github.com/mohameA337/churn-predection-mlops-project |

---

## Summary

This project demonstrates a **complete MLOps pipeline** from raw data to production:

1. ✅ **Data Validation:** Schema + null checks + distribution analysis
2. ✅ **Preprocessing:** ColumnTransformer (StandardScaler + OneHotEncoder) fitted on train only
3. ✅ **Experiment Tracking:** MLflow logs 6 models (LR, DT, RF, XGBoost, LightGBM, RF+SMOTE)
4. ✅ **Model Selection:** Automatic best-model selection by validation ROC-AUC
5. ✅ **API:** FastAPI with Pydantic v2 input validation
6. ✅ **Containerization:** Docker image (python:3.11-slim, ~500MB)
7. ✅ **Cloud Deployment:** Render.com managed container platform
8. ✅ **Monitoring:** Prometheus + Grafana (9 panels: latency, throughput, error rate, CPU/memory, prediction distribution)
9. ✅ **Testing:** pytest unit + integration tests
10. ✅ **Reproducibility:** All code, configs, and artifacts in git

**Report:** See `reports/report.pdf` for detailed documentation of all 11 sections (introduction, schema, preparation, modeling, architecture, API design, deployment, monitoring, testing, conclusion, tools).
