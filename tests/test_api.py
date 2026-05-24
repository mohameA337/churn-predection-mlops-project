"""
API unit tests.
Run with: pytest tests/test_api.py -v

Requires the artifacts/ folder to exist (run the notebook first).
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))
from main import app

client = TestClient(app)

VALID_CUSTOMER = {
    "tenure": 24,
    "MonthlyCharges": 65.5,
    "TotalCharges": 1572.0,
    "SeniorCitizen": 0,
    "gender": "Female",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_predict_valid_input():
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "model_version" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["churn_prediction"], bool)


def test_predict_invalid_contract():
    bad = VALID_CUSTOMER.copy()
    bad["Contract"] = "Weekly"
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_negative_tenure():
    bad = VALID_CUSTOMER.copy()
    bad["tenure"] = -1
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_missing_field():
    bad = {k: v for k, v in VALID_CUSTOMER.items() if k != "MonthlyCharges"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
