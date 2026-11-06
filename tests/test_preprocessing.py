"""Tests for preprocessing module."""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing import scale_features, split_data, preprocess_pipeline
from src.config import FEATURE_COLS, SCALE_COLS


class TestScaleFeatures:
    """Test feature scaling."""

    def test_scale_returns_correct_shape(self, sample_dataframe):
        X = sample_dataframe[FEATURE_COLS]
        X_train, X_test = X[:160], X[160:]
        X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
        assert X_train_s.shape == X_train.shape
        assert X_test_s.shape == X_test.shape

    def test_scaled_columns_have_zero_mean(self, sample_dataframe):
        X = sample_dataframe[FEATURE_COLS]
        X_train, X_test = X[:160], X[160:]
        X_train_s, _, scaler = scale_features(X_train, X_test)
        for col in SCALE_COLS:
            assert abs(X_train_s[col].mean()) < 0.01

    def test_scaler_is_fitted(self, sample_dataframe):
        X = sample_dataframe[FEATURE_COLS]
        X_train, X_test = X[:160], X[160:]
        _, _, scaler = scale_features(X_train, X_test)
        assert hasattr(scaler, "mean_")


class TestSplitData:
    """Test train/test split."""

    def test_split_preserves_total(self, sample_dataframe):
        X_train, X_test, y_train, y_test = split_data(sample_dataframe)
        assert len(X_train) + len(X_test) == len(sample_dataframe)

    def test_stratified_split(self, sample_dataframe):
        _, _, y_train, y_test = split_data(sample_dataframe)
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.05


class TestPreprocessPipeline:
    """Test the full preprocessing pipeline."""

    def test_pipeline_returns_dict(self, sample_dataframe):
        result = preprocess_pipeline(sample_dataframe)
        assert "X_train" in result
        assert "X_test" in result
        assert "y_train" in result
        assert "y_test" in result
        assert "scaler" in result
