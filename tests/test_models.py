"""Tests for models module."""

import pytest
import tempfile
import joblib
from pathlib import Path

from src.models import (
    get_random_forest, get_xgboost, create_model_from_params,
    train_model, get_all_models
)


class TestModelCreation:
    """Test model factory functions."""

    def test_create_random_forest(self):
        model = get_random_forest()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_create_xgboost(self):
        model = get_xgboost()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_create_xgboost_with_weight(self):
        model = get_xgboost(scale_pos_weight=10.0)
        assert model.get_params()["scale_pos_weight"] == 10.0

    def test_create_from_params_rf(self):
        model = create_model_from_params("random_forest", {"n_estimators": 50})
        assert model.get_params()["n_estimators"] == 50

    def test_create_from_params_xgb(self):
        model = create_model_from_params("xgboost", {"n_estimators": 50})
        assert model.get_params()["n_estimators"] == 50

    def test_create_from_params_invalid(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model_from_params("invalid_model", {})

    def test_get_all_models(self):
        models = get_all_models()
        assert "Random Forest" in models
        assert "XGBoost" in models


class TestModelTraining:
    """Test model training."""

    def test_train_model(self, sample_features):
        X, y = sample_features
        model = get_random_forest({"n_estimators": 10, "max_depth": 3,
                                    "random_state": 42, "n_jobs": 1})
        trained, train_time = train_model(model, X, y, "test_model")
        assert train_time > 0
        assert hasattr(trained, "predict_proba")

    def test_trained_model_predicts(self, sample_features):
        X, y = sample_features
        model = get_random_forest({"n_estimators": 10, "max_depth": 3,
                                    "random_state": 42, "n_jobs": 1})
        trained, _ = train_model(model, X, y)
        predictions = trained.predict(X)
        assert len(predictions) == len(y)
        assert set(predictions).issubset({0, 1})
