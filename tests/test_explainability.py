"""Tests for SHAP explainability module."""

import pytest
import numpy as np
from src.config import FEATURE_COLS


class TestFraudExplainer:
    """Test SHAP explainability."""

    def test_explain_single(self, trained_rf_model, sample_transaction):
        """Test single prediction explanation."""
        try:
            from src.explainability import FraudExplainer
        except ImportError:
            pytest.skip("SHAP not installed")

        explainer = FraudExplainer(trained_rf_model, model_name="test_rf")
        explanation = explainer.explain_single(sample_transaction)

        assert "base_value" in explanation
        assert "shap_values" in explanation
        assert "top_features" in explanation
        assert len(explanation["shap_values"]) == len(FEATURE_COLS)

    def test_compute_shap_values(self, trained_rf_model, sample_features):
        """Test batch SHAP value computation."""
        try:
            from src.explainability import FraudExplainer
        except ImportError:
            pytest.skip("SHAP not installed")

        X, _ = sample_features
        explainer = FraudExplainer(trained_rf_model, model_name="test_rf")
        shap_values = explainer.compute_shap_values(X, max_samples=20)

        assert shap_values.shape[1] == len(FEATURE_COLS)

    def test_top_features_sorted(self, trained_rf_model, sample_transaction):
        """Top features should be sorted by absolute SHAP value."""
        try:
            from src.explainability import FraudExplainer
        except ImportError:
            pytest.skip("SHAP not installed")

        explainer = FraudExplainer(trained_rf_model, model_name="test_rf")
        explanation = explainer.explain_single(sample_transaction)

        top = explanation["top_features"]
        for i in range(len(top) - 1):
            assert top[i]["abs_shap_value"] >= top[i + 1]["abs_shap_value"]
