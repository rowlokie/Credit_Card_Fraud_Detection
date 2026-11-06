"""
Class Balancing Module
======================
Handles class imbalance using SMOTE (Synthetic Minority Oversampling Technique).
CRITICAL: SMOTE is applied ONLY to the training set.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from src.config import RANDOM_STATE, SMOTE_SAMPLING_STRATEGY, SMOTE_K_NEIGHBORS
from src.logging_config import get_logger

logger = get_logger(__name__)


def apply_smote(X_train, y_train, sampling_strategy=None, k_neighbors=None):
    """
    Apply SMOTE oversampling to balance the training set.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training target labels.
    sampling_strategy : float, optional
        Desired ratio of minority to majority samples after resampling.
    k_neighbors : int, optional
        Number of nearest neighbors for SMOTE.

    Returns
    -------
    tuple : (X_resampled, y_resampled)
    """
    sampling_strategy = sampling_strategy or SMOTE_SAMPLING_STRATEGY
    k_neighbors = k_neighbors or SMOTE_K_NEIGHBORS

    before_counts = pd.Series(y_train).value_counts()
    logger.info("⚖️  CLASS BALANCING (SMOTE)")
    logger.info("─" * 40)
    logger.info("   Before SMOTE:")
    logger.info("     Legitimate (0): %8s", f"{before_counts.get(0, 0):,}")
    logger.info("     Fraudulent (1): %8s", f"{before_counts.get(1, 0):,}")

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=RANDOM_STATE,
    )

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    after_counts = pd.Series(y_resampled).value_counts()
    logger.info("   After SMOTE:")
    logger.info("     Legitimate (0): %8s", f"{after_counts.get(0, 0):,}")
    logger.info("     Fraudulent (1): %8s", f"{after_counts.get(1, 0):,}")

    synthetic_samples = len(y_resampled) - len(y_train)
    logger.info("   ✅ Generated %s synthetic fraud samples", f"{synthetic_samples:,}")

    return X_resampled, y_resampled


def get_class_distribution(y):
    """Get the class distribution of a target vector."""
    counts = pd.Series(y).value_counts().sort_index()
    total = len(y)

    return {
        "counts": counts.to_dict(),
        "percentages": {k: v / total * 100 for k, v in counts.items()},
        "total": total,
        "ratio": counts.get(0, 0) / max(counts.get(1, 0), 1),
    }
