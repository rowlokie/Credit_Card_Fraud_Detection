"""
Shared pytest fixtures for credit card fraud detection tests.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import FEATURE_COLS, TARGET_COL, RANDOM_STATE


@pytest.fixture
def sample_dataframe():
    """Create a small synthetic fraud dataset for testing."""
    np.random.seed(RANDOM_STATE)
    n_samples = 200
    n_fraud = 10

    data = {}
    data["Time"] = np.random.uniform(0, 172800, n_samples)
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n_samples)
    data["Amount"] = np.abs(np.random.lognormal(3, 2, n_samples))

    labels = np.zeros(n_samples, dtype=int)
    labels[:n_fraud] = 1
    np.random.shuffle(labels)
    data["Class"] = labels

    return pd.DataFrame(data)


@pytest.fixture
def sample_features(sample_dataframe):
    """Extract features and target from sample dataframe."""
    X = sample_dataframe[FEATURE_COLS]
    y = sample_dataframe[TARGET_COL]
    return X, y


@pytest.fixture
def trained_rf_model(sample_features):
    """A pre-fitted lightweight Random Forest model."""
    X, y = sample_features
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=RANDOM_STATE)
    model.fit(X, y)
    return model


@pytest.fixture
def sample_transaction():
    """A single transaction dict for prediction testing."""
    return {
        "Time": 406.0,
        "V1": -0.24, "V2": 0.70, "V3": 0.53, "V4": 0.24,
        "V5": -0.28, "V6": -0.17, "V7": -0.05, "V8": 0.09,
        "V9": 0.23, "V10": -0.11, "V11": -0.63, "V12": -0.08,
        "V13": -0.31, "V14": 0.39, "V15": -0.45, "V16": 0.09,
        "V17": 0.27, "V18": -0.08, "V19": -0.09, "V20": -0.07,
        "V21": 0.10, "V22": 0.30, "V23": -0.01, "V24": 0.02,
        "V25": 0.31, "V26": 0.04, "V27": 0.04, "V28": 0.01,
        "Amount": 44.50,
    }


@pytest.fixture
def fraud_transaction():
    """A sample fraud transaction dict."""
    return {
        "Time": 80000.0,
        "V1": -3.04, "V2": -3.16, "V3": 1.09, "V4": 2.29,
        "V5": 1.36, "V6": -1.06, "V7": -2.74, "V8": 1.01,
        "V9": -0.68, "V10": -4.91, "V11": 3.93, "V12": -5.41,
        "V13": 0.14, "V14": -8.10, "V15": 0.30, "V16": -4.76,
        "V17": -7.43, "V18": -2.55, "V19": 0.69, "V20": 0.58,
        "V21": 0.53, "V22": 0.22, "V23": -0.15, "V24": -0.40,
        "V25": 0.15, "V26": -0.21, "V27": 0.73, "V28": 0.44,
        "Amount": 1.98,
    }
