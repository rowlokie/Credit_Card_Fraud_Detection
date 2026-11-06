"""Tests for evaluation module."""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.evaluation import evaluate_model, compare_models, get_best_model


class TestEvaluateModel:
    """Test model evaluation."""

    def test_evaluate_returns_metrics(self, sample_features, trained_rf_model):
        X, y = sample_features
        metrics = evaluate_model(trained_rf_model, X, y, model_name="test_rf")
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "auc_roc" in metrics
        assert "confusion_matrix" in metrics

    def test_metrics_in_range(self, sample_features, trained_rf_model):
        X, y = sample_features
        metrics = evaluate_model(trained_rf_model, X, y)
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1

    def test_custom_threshold(self, sample_features, trained_rf_model):
        X, y = sample_features
        metrics_default = evaluate_model(trained_rf_model, X, y)
        metrics_low = evaluate_model(trained_rf_model, X, y, threshold=0.1)
        # Lower threshold should give higher recall
        assert metrics_low["recall"] >= metrics_default["recall"] - 0.01


class TestCompareModels:
    """Test model comparison."""

    def test_compare_returns_dataframe(self, sample_features, trained_rf_model):
        X, y = sample_features
        m1 = evaluate_model(trained_rf_model, X, y, model_name="Model_A")
        m2 = evaluate_model(trained_rf_model, X, y, model_name="Model_B")
        df = compare_models([m1, m2])
        assert len(df) == 2

    def test_get_best_model(self, sample_features, trained_rf_model):
        X, y = sample_features
        m1 = evaluate_model(trained_rf_model, X, y, model_name="A")
        m2 = evaluate_model(trained_rf_model, X, y, model_name="B")
        best = get_best_model([m1, m2], metric="f1_score")
        assert "model_name" in best
