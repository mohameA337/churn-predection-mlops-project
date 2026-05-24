"""
Preprocessing unit tests.
Run with: pytest tests/test_preprocess.py -v

Requires artifacts/preprocessor.pkl to exist (run the notebook first).
"""
import pytest
import joblib
import numpy as np
import pandas as pd


SAMPLE_ROW = {
    "tenure": 12,
    "MonthlyCharges": 50.0,
    "TotalCharges": 600.0,
    "SeniorCitizen": 0,
    "gender": "Male",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Mailed check",
}

NUMERIC_FEATURES     = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
BINARY_FEATURES = ["SeniorCitizen"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


@pytest.fixture(scope="module")
def preprocessor():
    return joblib.load("artifacts/preprocessor.pkl")


def test_preprocessor_loads(preprocessor):
    assert preprocessor is not None


def test_output_is_numeric(preprocessor):
    row = pd.DataFrame([SAMPLE_ROW])[ALL_FEATURES]
    result = preprocessor.transform(row)
    assert np.issubdtype(result.dtype, np.floating)


def test_output_has_no_nans(preprocessor):
    row = pd.DataFrame([SAMPLE_ROW])[ALL_FEATURES]
    result = preprocessor.transform(row)
    assert not np.isnan(result).any()


def test_output_shape(preprocessor):
    row = pd.DataFrame([SAMPLE_ROW])[ALL_FEATURES]
    result = preprocessor.transform(row)
    assert result.shape[0] == 1
    assert result.shape[1] > len(ALL_FEATURES)  # OHE expands categorical columns
