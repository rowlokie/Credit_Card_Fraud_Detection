"""
Models Module
=============
Defines, trains, and serializes machine learning models for fraud detection.
Supports creating models from optimized parameters.
"""

import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import (
    RANDOM_FOREST_PARAMS, XGBOOST_PARAMS,
    RANDOM_STATE, get_model_path, get_scaler_path, get_metrics_path
)
from src.logging_config import get_logger

logger = get_logger(__name__)


def get_random_forest(params=None):
    """Create a configured Random Forest classifier."""
    p = params or RANDOM_FOREST_PARAMS
    return RandomForestClassifier(**p)


def get_xgboost(scale_pos_weight=None, params=None):
    """Create a configured XGBoost classifier."""
    p = (params or XGBOOST_PARAMS).copy()
    if scale_pos_weight is not None:
        p["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**p)


def create_model_from_params(model_type, params, scale_pos_weight=None):
    """
    Factory function to create a model from optimized hyperparameters.

    Parameters
    ----------
    model_type : str
        "random_forest" or "xgboost"
    params : dict
        Hyperparameters from optimizer.
    scale_pos_weight : float, optional
        Class weight ratio for XGBoost.

    Returns
    -------
    estimator
        Unfitted model instance.
    """
    if model_type == "random_forest":
        full_params = {**RANDOM_FOREST_PARAMS, **params}
        full_params["random_state"] = RANDOM_STATE
        return RandomForestClassifier(**full_params)
    elif model_type == "xgboost":
        full_params = {**XGBOOST_PARAMS, **params}
        full_params["random_state"] = RANDOM_STATE
        if scale_pos_weight is not None:
            full_params["scale_pos_weight"] = scale_pos_weight
        return XGBClassifier(**full_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(model, X_train, y_train, model_name="model"):
    """Train a model and measure training time."""
    logger.info("🏋️  Training %s...", model_name)
    logger.info("   Training samples: %s", f"{len(y_train):,}")

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    logger.info("   ✅ %s trained in %.2fs", model_name, training_time)
    return model, training_time


def get_all_models(scale_pos_weight=None):
    """Get a dictionary of all models to train."""
    return {
        "Random Forest": get_random_forest(),
        "XGBoost": get_xgboost(scale_pos_weight=scale_pos_weight),
    }


def save_model(model, model_name, scaler=None, metrics=None):
    """Save a trained model (and optionally scaler and metrics) to disk."""
    model_path = get_model_path(model_name.lower().replace(" ", "_"))
    joblib.dump(model, model_path)
    logger.info("   💾 Model saved: %s", model_path)

    if scaler is not None:
        scaler_path = get_scaler_path()
        joblib.dump(scaler, scaler_path)
        logger.info("   💾 Scaler saved: %s", scaler_path)

    if metrics is not None:
        metrics_path = get_metrics_path()
        joblib.dump(metrics, metrics_path)
        logger.info("   💾 Metrics saved: %s", metrics_path)


def load_model(model_name):
    """Load a trained model from disk."""
    model_path = get_model_path(model_name.lower().replace(" ", "_"))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run 'python train.py' first."
        )
    return joblib.load(model_path)


def load_scaler():
    """Load the fitted scaler from disk."""
    scaler_path = get_scaler_path()
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. Run 'python train.py' first."
        )
    return joblib.load(scaler_path)


def load_metrics():
    """Load saved training metrics from disk."""
    metrics_path = get_metrics_path()
    if not metrics_path.exists():
        return None
    return joblib.load(metrics_path)
