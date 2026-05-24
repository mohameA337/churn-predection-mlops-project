import os
import joblib
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# Resolve artifacts folder: same dir as script (Docker) or one level up (local dev)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ARTIFACTS  = (os.path.join(_SCRIPT_DIR, "artifacts")
               if os.path.isdir(os.path.join(_SCRIPT_DIR, "artifacts"))
               else os.path.join(_SCRIPT_DIR, "..", "artifacts"))

# ---------- Load artifacts once at startup ----------
preprocessor = joblib.load(os.path.join(_ARTIFACTS, "preprocessor.pkl"))
model        = joblib.load(os.path.join(_ARTIFACTS, "model.pkl"))

with open(os.path.join(_ARTIFACTS, "schema.json")) as f:
    schema = json.load(f)

MODEL_VERSION = schema["model_version"]
ALL_FEATURES  = schema["all_features"]

# ---------- Custom Prometheus metrics ----------
PREDICTION_COUNTER = Counter(
    "churn_predictions_total",
    "Total predictions made",
    ["prediction"]
)
PROBA_HISTOGRAM = Histogram(
    "churn_probability",
    "Distribution of predicted churn probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# ---------- App ----------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a telecom customer is likely to churn.",
    version="1.0.0",
)

Instrumentator().instrument(app).expose(app)


# ---------- Input schema (Pydantic v2) ----------
class CustomerRecord(BaseModel):
    tenure:           int   = Field(..., ge=0,  description="Months with company")
    MonthlyCharges:   float = Field(..., gt=0,  description="Monthly bill in USD")
    TotalCharges:     float = Field(..., ge=0,  description="Total amount billed")
    SeniorCitizen:    int   = Field(..., ge=0, le=1)
    gender:           str
    Partner:          str
    Dependents:       str
    PhoneService:     str
    MultipleLines:    str
    InternetService:  str
    OnlineSecurity:   str
    OnlineBackup:     str
    DeviceProtection: str
    TechSupport:      str
    StreamingTV:      str
    StreamingMovies:  str
    Contract:         str
    PaperlessBilling: str
    PaymentMethod:    str

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        allowed = {"Month-to-month", "One year", "Two year"}
        if v not in allowed:
            raise ValueError(f"Contract must be one of {allowed}")
        return v


# ---------- Response schema ----------
class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction:  bool
    model_version:     str


# ---------- Endpoints ----------
@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse, summary="Predict churn for one customer")
def predict(customer: CustomerRecord):
    try:
        row      = pd.DataFrame([customer.model_dump()])[ALL_FEATURES]
        row_proc = preprocessor.transform(row)
        prob     = float(model.predict_proba(row_proc)[0, 1])
        pred     = bool(prob >= 0.5)

        PREDICTION_COUNTER.labels(prediction=str(pred)).inc()
        PROBA_HISTOGRAM.observe(prob)

        return PredictionResponse(
            churn_probability=round(prob, 4),
            churn_prediction=pred,
            model_version=MODEL_VERSION,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
