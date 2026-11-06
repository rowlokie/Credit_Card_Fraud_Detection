"""
Preprocessing Module
====================
Handles feature scaling, optional PCA dimensionality reduction,
and train/test splitting for the fraud detection pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_STATE, TEST_SIZE, TARGET_COL,
    FEATURE_COLS, SCALE_COLS, N_PCA_COMPONENTS
)
from src.logging_config import get_logger

logger = get_logger(__name__)


def scale_features(X_train, X_test, columns=None):
    """
    Apply StandardScaler to specified columns.
    Fitted on training data only to prevent data leakage.
    """
    columns = columns or SCALE_COLS
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[columns] = scaler.fit_transform(X_train[columns])
    X_test_scaled[columns] = scaler.transform(X_test[columns])

    logger.info("   ✅ Scaled features: %s", columns)
    return X_train_scaled, X_test_scaled, scaler


def apply_pca(X_train, X_test, n_components=None):
    """Apply PCA dimensionality reduction."""
    n_components = n_components or N_PCA_COMPONENTS

    if n_components is None:
        logger.info("   ℹ️  PCA skipped (V1-V28 are already PCA-transformed)")
        return X_train, X_test, None

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    explained_var = sum(pca.explained_variance_ratio_) * 100
    logger.info("   ✅ PCA: %d features → %d components (%.1f%% variance)",
                X_train.shape[1], n_components, explained_var)

    return X_train_pca, X_test_pca, pca


def split_data(df, test_size=None):
    """Split dataset into training and test sets with stratified sampling."""
    test_size = test_size or TEST_SIZE

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    train_fraud = y_train.sum()
    test_fraud = y_test.sum()
    logger.info("📊 Data Split (%.0f%% train / %.0f%% test):",
                (1 - test_size) * 100, test_size * 100)
    logger.info("   Train: %s samples (%s fraud, %.3f%%)",
                f"{len(X_train):,}", f"{train_fraud:,}", train_fraud / len(y_train) * 100)
    logger.info("   Test:  %s samples (%s fraud, %.3f%%)",
                f"{len(X_test):,}", f"{test_fraud:,}", test_fraud / len(y_test) * 100)

    return X_train, X_test, y_train, y_test


def preprocess_pipeline(df, n_pca_components=None):
    """Run the full preprocessing pipeline: split → scale → (optional PCA)."""
    logger.info("🔧 PREPROCESSING PIPELINE")
    logger.info("─" * 40)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test, scaler = scale_features(X_train, X_test)
    X_train, X_test, pca = apply_pca(X_train, X_test, n_pca_components)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "pca": pca,
    }
