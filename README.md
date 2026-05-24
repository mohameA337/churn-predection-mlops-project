# Customer Churn Prediction — MLOps Project

End-to-end ML service that predicts telecom customer churn. Built for an undergraduate MLOps course.

---

## Setup

```bash
git clone https://github.com/mohameA337/churn-predection-mlops-project.git
cd churn-mlops
pip install -r api/requirements.txt
```

Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root.

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

The API is deployed publicly. Test with:
```bash
curl https://<your-render-url>/health
curl -X POST https://<your-render-url>/predict -H "Content-Type: application/json" -d '{...}'
```

---

## Kubernetes Deployment

```bash
# 1. Push image to Docker Hub (already pushed as mayoubo/churn-api:latest)
docker tag churn-api mayoubo/churn-api:latest
docker push mayoubo/churn-api:latest

# 2. Apply to cluster
kubectl apply -f deployment/k8s-deployment.yaml

# 3. Get public IP
kubectl get service churn-api-service
```

Config: 2-replica Deployment + LoadBalancer Service + Nginx Ingress.

---

## Experiment Results

| Model | Val ROC-AUC | Val F1 | Val Recall |
|---|---|---|---|
| Logistic Regression | 0.8291 | 0.6136 | 0.7687 |
| Decision Tree | 0.8089 | 0.5857 | 0.7722 |
| **Random Forest** | **0.8344** | **0.6368** | 0.7082 |
| XGBoost | 0.8249 | 0.6021 | 0.7189 |
| LightGBM | 0.8179 | 0.6109 | 0.7011 |
| RF + SMOTE | *(see experiment_log.csv)* | | |

**Final model: Random Forest** (rf-v1) — highest validation ROC-AUC and F1.  
Class imbalance (26.5% churn) handled with `class_weight='balanced'` and SMOTE over-sampling comparison.  
Full run details: `experiments/experiment_log.csv` or `mlflow ui`.

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
