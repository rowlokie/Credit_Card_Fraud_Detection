"""Tests for threshold optimization module."""

import pytest
import numpy as np
from src.threshold import ThresholdOptimizer


class TestThresholdOptimizer:
    """Test threshold optimization strategies."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample true labels and predicted probabilities."""
        np.random.seed(42)
        y_true = np.array([0]*90 + [1]*10)
        y_proba = np.random.uniform(0, 0.3, 90).tolist() + np.random.uniform(0.5, 1.0, 10).tolist()
        return np.array(y_true), np.array(y_proba)

    def test_f1_optimization(self, sample_predictions):
        y_true, y_proba = sample_predictions
        optimizer = ThresholdOptimizer(strategy="f1")
        result = optimizer.find_optimal(y_true, y_proba)
        assert 0 < result["optimal_threshold"] < 1
        assert result["strategy"] == "f1"

    def test_youden_optimization(self, sample_predictions):
        y_true, y_proba = sample_predictions
        optimizer = ThresholdOptimizer(strategy="youden")
        result = optimizer.find_optimal(y_true, y_proba)
        assert 0 < result["optimal_threshold"] < 1

    def test_cost_optimization(self, sample_predictions):
        y_true, y_proba = sample_predictions
        optimizer = ThresholdOptimizer(strategy="cost", cost_fp=1.0, cost_fn=10.0)
        result = optimizer.find_optimal(y_true, y_proba)
        assert 0 < result["optimal_threshold"] < 1

    def test_pr_breakeven(self, sample_predictions):
        y_true, y_proba = sample_predictions
        optimizer = ThresholdOptimizer(strategy="precision_recall")
        result = optimizer.find_optimal(y_true, y_proba)
        assert 0 < result["optimal_threshold"] < 1

    def test_apply_threshold(self, sample_predictions):
        y_true, y_proba = sample_predictions
        optimizer = ThresholdOptimizer()
        optimizer.optimal_threshold = 0.5
        preds = optimizer.apply_threshold(y_proba)
        assert set(preds).issubset({0, 1})
        assert len(preds) == len(y_proba)

    def test_all_strategies_computed(self, sample_predictions):
        y_true, y_proba = sample_predictions
        optimizer = ThresholdOptimizer(strategy="f1")
        result = optimizer.find_optimal(y_true, y_proba)
        assert "all_strategies" in result
        assert "f1" in result["all_strategies"]
        assert "youden" in result["all_strategies"]
        assert "cost" in result["all_strategies"]
