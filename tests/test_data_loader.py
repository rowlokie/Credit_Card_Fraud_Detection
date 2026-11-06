"""Tests for data_loader module."""

import pytest
import pandas as pd
from src.data_loader import _validate_dataset
from src.config import FEATURE_COLS, TARGET_COL


class TestValidateDataset:
    """Test dataset validation logic."""

    def test_valid_dataset(self, sample_dataframe):
        """Valid dataset should pass without error."""
        _validate_dataset(sample_dataframe)

    def test_missing_columns(self, sample_dataframe):
        """Should raise ValueError for missing columns."""
        df = sample_dataframe.drop(columns=["V1", "V2"])
        with pytest.raises(ValueError, match="Missing required columns"):
            _validate_dataset(df)

    def test_invalid_target_values(self, sample_dataframe):
        """Should raise ValueError for non-binary target."""
        df = sample_dataframe.copy()
        df["Class"] = 5
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            _validate_dataset(df)

    def test_null_values_warning(self, sample_dataframe):
        """Should warn about null values."""
        df = sample_dataframe.copy()
        df.loc[0, "V1"] = None
        # Should not raise, but log warning
        _validate_dataset(df)
