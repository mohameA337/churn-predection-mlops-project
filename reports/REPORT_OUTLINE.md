# Final Report Outline — Customer Churn Prediction MLOps Project

Convert this outline to a PDF. Each section maps to the grading rubric.

---

## 1. Introduction

- **Business problem:** Telecom companies lose revenue when customers churn. The goal is to predict churn risk per customer so a retention team can intervene early.
- **Dataset:** IBM Telco Customer Churn — 7,043 customers, 21 columns, binary target (`Churn`: Yes / No)
- **Class distribution:** 73.5% No / 26.5% Yes — moderate imbalance handled with `class_weight='balanced'` and SMOTE over-sampling
- **Main objective:** Build a production-ready ML service emphasizing reproducibility, containerization, deployment, and monitoring — not just accuracy.

---

## 2. Dataset Schema & Assumptions

| Column | Type | Description |
|---|---|---|
| tenure | int | Months with company (0–72) |
| MonthlyCharges | float | Monthly bill in USD |
| TotalCharges | float | Total billed (11 whitespace entries → median imputed) |
| SeniorCitizen | int | Binary: 1 = senior citizen |
| Contract | str | Month-to-month / One year / Two year |
| InternetService | str | DSL / Fiber optic / No |
| PaymentMethod | str | Electronic check / Mailed check / Bank transfer / Credit card |
| Churn | int | **Target** — 1 = churned, 0 = stayed |

**Splits:** 70% train / 15% validation / 15% test — stratified on Churn.  
**Exclusions:** `customerID` dropped (identifier, no predictive value).  
**Outliers:** IQR analysis confirmed all numeric values are domain-valid billing figures (0 outliers). No rows dropped; StandardScaler reduces scale influence.

---

## 3. Data Preparation & Validation

- **Schema validation:** 21 required columns verified on load
- **Null checks:** 0 nulls in critical columns; 11 TotalCharges whitespace rows → coerced to float and median-imputed
- **Target distribution check:** 26.5% churn rate confirmed; no extreme imbalance flag triggered
- **Preprocessing pipeline:** `ColumnTransformer` combining:
  - `StandardScaler` for numeric features (tenure, MonthlyCharges, TotalCharges)
  - `OneHotEncoder` for 15 categorical features → 45 total processed features
  - Passthrough for `SeniorCitizen`
- **No data leakage:** pipeline fitted on training set only; validation and test sets transformed using the fitted pipeline

---

## 4. Modeling & Experiments

All runs logged to MLflow (`mlflow ui` → http://localhost:5000).

### 4.1 Baseline Models

| Model | Val ROC-AUC | Val F1 | Val Recall | Val Precision |
|---|---|---|---|---|
| Logistic Regression | 0.8291 | 0.6136 | 0.7687 | 0.5143 |
| Decision Tree | 0.8089 | 0.5857 | 0.7722 | 0.4730 |

### 4.2 Improved Models

| Model | Val ROC-AUC | Val F1 | Val Recall | Val Precision | Val Accuracy |
|---|---|---|---|---|---|
| **Random Forest** | **0.8344** | **0.6368** | 0.7082 | 0.5785 | 0.7852 |
| RF + SMOTE | 0.8333 | 0.6295 | 0.6833 | 0.5836 | 0.7862 |
| XGBoost | 0.8249 | 0.6021 | 0.7189 | 0.5179 | 0.7474 |
| LightGBM | 0.8179 | 0.6109 | 0.7011 | 0.5412 | 0.7625 |

### 4.3 Imbalance Handling

Two techniques were compared:
- `class_weight='balanced'` — adjusts the loss function to penalize minority-class errors more heavily (used in all models)
- **SMOTE** (Synthetic Minority Over-sampling Technique) — generates synthetic minority-class samples applied to `X_train_proc` *after* preprocessing to prevent data leakage

The final model is chosen automatically by validation ROC-AUC between Random Forest and RF+SMOTE.

### 4.4 Final Model: Random Forest — Justification

| Reason | Detail |
|---|---|
| Best Validation ROC-AUC | **0.8344** — highest among all 5 models |
| Best Validation F1 | **0.6368** — best precision-recall balance under class imbalance |
| SMOTE comparison | RF and RF+SMOTE both trained; final selected by val_roc_auc |
| Interpretable | Feature importance readable directly from the ensemble |
| vs LightGBM | +1.65 pp ROC-AUC, +2.59 pp F1 on validation |
| vs Logistic Regression | +0.53 pp ROC-AUC, +2.32 pp F1 |

### 4.5 Test Set Results (Random Forest, rf-v1)

| Metric | Value |
|---|---|
| Accuracy | 0.7947 |
| Precision | 0.5785 |
| Recall | 0.7082 |
| F1-score | 0.6368 |
| **ROC-AUC** | **0.8434** |

**Interpretation:** ROC-AUC of 0.8434 is solid for telecom churn (industry benchmark: 0.80+ = good). Precision of 0.58 is expected with 26.5% class imbalance when recall is prioritized — catching 71% of actual churners early is more valuable to the business than minimizing false positives. The model generalizes well: test ROC-AUC (0.8434) is slightly higher than validation (0.8344), confirming no overfitting.

---

## 5. MLOps Pipeline Architecture

```
Raw CSV (7,043 rows, 21 columns)
   │
   ▼
Data Validation (schema + null + distribution checks)
   │
   ▼
Preprocessing Pipeline (ColumnTransformer — fitted on train only)
   │  StandardScaler + OneHotEncoder + passthrough → 45 features
   ▼
Model Training × 6 runs — all logged to MLflow
   │  Logistic Regression, Decision Tree, Random Forest,
   │  XGBoost, LightGBM, RF+SMOTE
   ▼
Best model selected by val_roc_auc (Random Forest or RF+SMOTE)
   │
   ▼
Artifacts: model.pkl + preprocessor.pkl + schema.json
   │
   ▼
FastAPI Service  (POST /predict · GET /health · GET /metrics)
   │
   ▼
Docker Container  (python:3.11-slim, ~500 MB)
   │
   ├── Cloud VM — Render.com  (docker run -p 80:8000)
   │     publicly accessible at https://<render-url>
   └── Kubernetes — 2 replicas + LoadBalancer + Nginx Ingress
         │
         ▼
   Prometheus (scrapes /metrics every 15s)
         │
         ▼
   Grafana Dashboard (9 panels — latency, throughput, error rate,
                      CPU/memory, prediction distribution)
```

---

## 6. API Design

**Base URL (local):** `http://localhost:8001`  
**Base URL (cloud):** `https://<render-url>`

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns `{"status": "ok", "model_version": "rf-v1"}` |
| `/predict` | POST | Accepts customer JSON, returns churn probability + prediction |
| `/metrics` | GET | Prometheus-format metrics (scraped by Prometheus) |
| `/docs` | GET | Interactive Swagger documentation (auto-generated by FastAPI) |

**Example request/response:** see README.md.

**Input validation (Pydantic v2):**
- `tenure ≥ 0`, `MonthlyCharges > 0`, `TotalCharges ≥ 0`
- `Contract` must be one of: `Month-to-month`, `One year`, `Two year`
- All 19 fields required — missing fields return HTTP 422

---

## 7. Deployment

| Layer | Tool | Detail |
|---|---|---|
| Containerization | Docker | `python:3.11-slim`; artifacts copied at build time; libgomp1 installed for LightGBM compat |
| Local stack | docker-compose | API + Prometheus + Grafana; Prometheus datasource auto-provisioned |
| Cloud | Render.com | Free-tier web service; Docker image from Docker Hub (`mayoubo/churn-api:latest`) |
| Orchestration | Kubernetes | 2-replica Deployment + LoadBalancer Service + Nginx Ingress; config in `deployment/k8s-deployment.yaml` |

---

## 8. Monitoring

Metrics tracked via Prometheus + Grafana (9-panel dashboard):

| Panel | Metric | What it shows |
|---|---|---|
| Request Rate | `http_requests_total` | Requests per second (stat) |
| Error Rate | `http_requests_total{status=~"5.."}` | % of 5xx errors |
| Latency p95 | `http_request_duration_seconds_bucket` | 95th percentile response time |
| Prediction Count | `churn_predictions_total` | True vs False predictions (model view) |
| Request Rate Over Time | — | Throughput trend (time series) |
| Latency Percentiles | — | p50 / p95 / p99 over time |
| Probability Distribution | `churn_probability_bucket` | Model-level drift indicator |
| CPU Usage | `process_cpu_seconds_total` | API process CPU % |
| Memory Usage | `process_resident_memory_bytes` | API RSS memory (MB) |

Grafana dashboard at `http://localhost:3000` (admin/admin) — starts automatically with `docker-compose up`.

---

## 9. Testing

```bash
pytest tests/ -v
```

| Test file | Tests |
|---|---|
| `tests/test_api.py` | Health check, valid predict, invalid contract (422), negative tenure (422), missing field (422) |
| `tests/test_preprocess.py` | Preprocessor loads, output is float, no NaNs, correct shape |

---

## 10. Conclusion

- Built a complete MLOps pipeline: data validation → training → experiment tracking → API → containerization → cloud deployment → monitoring
- **Random Forest (rf-v1)** selected as final model with ROC-AUC **0.8344** on validation — highest among all 6 trained models
- Two imbalance strategies compared: `class_weight='balanced'` and SMOTE; winner selected automatically
- All experiments reproducible via MLflow; all infrastructure reproducible via Docker and Kubernetes configs
- System is publicly accessible on Render.com; any client can call `/predict`

---

## 11. Tools Used

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11 | Core language (Docker) / 3.14 (local dev) |
| pandas, scikit-learn | latest | Data processing and modeling |
| imbalanced-learn | latest | SMOTE over-sampling |
| XGBoost, LightGBM | latest | Boosting baselines |
| **Random Forest** | (scikit-learn) | **Production model** |
| MLflow | latest | Experiment tracking and model registry |
| FastAPI + Uvicorn | latest | Prediction API |
| Docker + docker-compose | latest | Containerization and local stack |
| Kubernetes | — | Orchestrated deployment (k8s-deployment.yaml) |
| Render.com | — | Cloud deployment |
| Prometheus + Grafana | 2.52 / 10.4.2 | Monitoring and dashboards |
| pytest | latest | Unit and integration testing |
