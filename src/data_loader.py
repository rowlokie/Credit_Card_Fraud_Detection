"""
Data Loader Module
==================
Handles loading and validating the Credit Card Fraud Detection dataset.
Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

import sys
import pandas as pd
import numpy as np

from src.config import RAW_DATA_PATH, TARGET_COL, FEATURE_COLS
from src.logging_config import get_logger

logger = get_logger(__name__)


def load_dataset(filepath=None):
    """
    Load the credit card fraud detection dataset from CSV.

    Parameters
    ----------
    filepath : str or Path, optional
        Path to the CSV file. Defaults to config.RAW_DATA_PATH.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    ValueError
        If the dataset fails validation checks.
    """
    filepath = filepath or RAW_DATA_PATH

    if not filepath.exists():
        logger.error("Dataset not found at %s", filepath)
        logger.error("Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        logger.error("Place 'creditcard.csv' in: %s", filepath.parent)
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    logger.info("📂 Loading dataset from: %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("   ✅ Loaded %s transactions", f"{len(df):,}")

    _validate_dataset(df)
    return df


def _validate_dataset(df):
    """Validate the dataset schema and integrity."""
    required_cols = FEATURE_COLS + [TARGET_COL]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        logger.warning("Found null values:\n%s", null_counts[null_counts > 0])
    else:
        logger.info("   ✅ No missing values detected")

    unique_classes = df[TARGET_COL].unique()
    if not set(unique_classes).issubset({0, 1}):
        raise ValueError(f"Target column must contain only 0 and 1, found: {unique_classes}")

    logger.info("   ✅ Dataset validation passed")


def get_dataset_summary(df):
    """
    Generate and log a comprehensive summary of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to summarize.

    Returns
    -------
    dict
        Summary statistics.
    """
    total = len(df)
    fraud = df[TARGET_COL].sum()
    legit = total - fraud
    fraud_pct = (fraud / total) * 100

    summary = {
        "total_transactions": total,
        "legitimate": int(legit),
        "fraudulent": int(fraud),
        "fraud_percentage": fraud_pct,
        "n_features": len(FEATURE_COLS),
        "amount_min": df["Amount"].min(),
        "amount_max": df["Amount"].max(),
        "amount_mean": df["Amount"].mean(),
        "amount_median": df["Amount"].median(),
        "time_span_hours": df["Time"].max() / 3600,
    }

    logger.info("═" * 60)
    logger.info("  DATASET SUMMARY")
    logger.info("═" * 60)
    logger.info("  Total Transactions:    %10s", f"{total:,}")
    logger.info("  Legitimate (Class 0):  %10s  (%.3f%%)", f"{legit:,}", 100 - fraud_pct)
    logger.info("  Fraudulent (Class 1):  %10s  (%.3f%%)", f"{fraud:,}", fraud_pct)
    logger.info("  Features:              %10d", len(FEATURE_COLS))
    logger.info("  Amount Range:          $%.2f - $%.2f", summary["amount_min"], summary["amount_max"])
    logger.info("═" * 60)

    return summary
