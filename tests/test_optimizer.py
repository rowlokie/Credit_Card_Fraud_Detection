"""Tests for optimizer module."""

import pytest
import numpy as np
from src.optimizer import cross_validate_model
from sklearn.ensemble import RandomForestClassifier


class TestCrossValidation:
    """Test cross-validation functionality."""

    def test_cv_returns_scores(self, sample_features):
        X, y = sample_features
        model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
        result = cross_validate_model(model, X, y, cv_folds=3, scoring="f1")
        assert "scores" in result
        assert "mean" in result
        assert "std" in result
        assert len(result["scores"]) == 3

    def test_cv_scores_in_range(self, sample_features):
        X, y = sample_features
        model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
        result = cross_validate_model(model, X, y, cv_folds=3)
        assert 0 <= result["mean"] <= 1
        assert result["std"] >= 0
