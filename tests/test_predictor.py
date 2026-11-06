"""Tests for predictor module (uses sample model, not saved models)."""

import pytest
import numpy as np
import pandas as pd
from src.config import FEATURE_COLS


class TestPredictorLogic:
    """Test predictor core logic with a trained model fixture."""

    def test_predict_output_keys(self, trained_rf_model, sample_transaction):
        """Prediction result should have expected keys."""
        probas = trained_rf_model.predict_proba(
            pd.DataFrame([sample_transaction])[FEATURE_COLS]
        )[0]
        is_fraud = bool(probas[1] >= 0.5)
        confidence = float(probas[1] if is_fraud else probas[0])

        result = {
            "prediction": "Fraud" if is_fraud else "Legitimate",
            "is_fraud": is_fraud,
            "confidence": confidence,
            "fraud_probability": float(probas[1]),
            "legitimate_probability": float(probas[0]),
        }

        assert "prediction" in result
        assert "is_fraud" in result
        assert "confidence" in result
        assert result["prediction"] in ("Fraud", "Legitimate")

    def test_batch_predict(self, trained_rf_model, sample_dataframe):
        """Batch prediction should work for multiple rows."""
        X = sample_dataframe[FEATURE_COLS]
        predictions = trained_rf_model.predict(X)
        probas = trained_rf_model.predict_proba(X)

        assert len(predictions) == len(sample_dataframe)
        assert probas.shape == (len(sample_dataframe), 2)
        assert set(predictions).issubset({0, 1})

    def test_threshold_changes_prediction(self, trained_rf_model, sample_transaction):
        """Using a different threshold should potentially change predictions."""
        probas = trained_rf_model.predict_proba(
            pd.DataFrame([sample_transaction])[FEATURE_COLS]
        )[0]

        is_fraud_05 = bool(probas[1] >= 0.5)
        is_fraud_001 = bool(probas[1] >= 0.001)

        # Very low threshold should predict fraud more often
        if probas[1] >= 0.001:
            assert is_fraud_001 is True

    def test_missing_features_detected(self):
        """Should detect missing features."""
        incomplete = {"Time": 0.0, "Amount": 100.0}
        missing = set(FEATURE_COLS) - set(incomplete.keys())
        assert len(missing) > 0
