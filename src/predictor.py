"""
Predictor Module
================
High-level FraudPredictor class with optimal threshold support and
SHAP explanation integration.
"""

import numpy as np
import pandas as pd
import joblib

from src.config import (
    FEATURE_COLS, SCALE_COLS,
    get_model_path, get_scaler_path, get_metrics_path,
    get_threshold_path, MODELS_DIR
)
from src.logging_config import get_logger

logger = get_logger(__name__)


class FraudPredictor:
    """
    Fraud prediction engine with threshold optimization and SHAP support.

    Usage
    -----
    >>> predictor = FraudPredictor(model_name="xgboost")
    >>> result = predictor.predict_single({"Time": 0, "Amount": 149.62, "V1": -1.35, ...})
    """

    def __init__(self, model_name="xgboost"):
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.metrics = None
        self.optimal_threshold = 0.5
        self._load_artifacts()

    def _load_artifacts(self):
        """Load model, scaler, metrics, and optimal threshold from disk."""
        # Load model
        model_path = get_model_path(self.model_name)
        if not model_path.exists():
            available = list(MODELS_DIR.glob("*.joblib"))
            model_files = [f.stem for f in available
                           if f.stem not in ("scaler", "metrics")
                           and not f.stem.endswith("_threshold")
                           and not f.stem.endswith("_shap_explainer")]
            raise FileNotFoundError(
                f"Model '{self.model_name}' not found at {model_path}.\n"
                f"Available models: {model_files}\n"
                f"Run 'python train.py' first to train and save models."
            )
        self.model = joblib.load(model_path)
        logger.debug("Loaded model: %s", model_path)

        # Load scaler
        scaler_path = get_scaler_path()
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

        # Load metrics
        metrics_path = get_metrics_path()
        if metrics_path.exists():
            self.metrics = joblib.load(metrics_path)

        # Load optimal threshold
        threshold_path = get_threshold_path(self.model_name)
        if threshold_path.exists():
            threshold_data = joblib.load(threshold_path)
            self.optimal_threshold = threshold_data.get("threshold", 0.5)
            logger.debug("Loaded optimal threshold: %.4f", self.optimal_threshold)

    def predict_single(self, transaction, threshold=None):
        """
        Predict whether a single transaction is fraudulent.

        Parameters
        ----------
        transaction : dict
            Dictionary with feature names as keys.
        threshold : float, optional
            Custom threshold override. If None, uses optimal threshold.

        Returns
        -------
        dict
            Prediction result with confidence and probabilities.
        """
        self._validate_input(transaction)
        t = threshold if threshold is not None else self.optimal_threshold

        df = pd.DataFrame([transaction])[FEATURE_COLS]

        if self.scaler is not None:
            df[SCALE_COLS] = self.scaler.transform(df[SCALE_COLS])

        probabilities = self.model.predict_proba(df)[0]
        is_fraud = bool(probabilities[1] >= t)
        confidence = float(probabilities[1] if is_fraud else probabilities[0])

        return {
            "prediction": "Fraud" if is_fraud else "Legitimate",
            "is_fraud": is_fraud,
            "confidence": confidence,
            "fraud_probability": float(probabilities[1]),
            "legitimate_probability": float(probabilities[0]),
            "threshold_used": t,
        }

    def predict_batch(self, df, threshold=None):
        """
        Predict fraud for a batch of transactions.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with feature columns.
        threshold : float, optional
            Custom threshold override.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with added prediction columns.
        """
        t = threshold if threshold is not None else self.optimal_threshold
        X = df[FEATURE_COLS].copy()

        if self.scaler is not None:
            X[SCALE_COLS] = self.scaler.transform(X[SCALE_COLS])

        probabilities = self.model.predict_proba(X)
        predictions = (probabilities[:, 1] >= t).astype(int)

        result = df.copy()
        result["Prediction"] = ["Fraud" if p == 1 else "Legitimate" for p in predictions]
        result["Is_Fraud"] = predictions.astype(bool)
        result["Fraud_Probability"] = probabilities[:, 1]
        result["Confidence"] = np.where(
            predictions == 1, probabilities[:, 1], probabilities[:, 0]
        )
        result["Threshold_Used"] = t

        return result

    def _validate_input(self, transaction):
        """Validate that all required features are present."""
        missing = set(FEATURE_COLS) - set(transaction.keys())
        if missing:
            raise ValueError(
                f"Missing required features: {sorted(missing)}\n"
                f"Required features: {FEATURE_COLS}"
            )

    def get_model_info(self):
        """Get information about the loaded model."""
        info = {
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "has_scaler": self.scaler is not None,
            "features": FEATURE_COLS,
            "n_features": len(FEATURE_COLS),
            "optimal_threshold": self.optimal_threshold,
        }
        if self.metrics:
            info["training_metrics"] = self.metrics
        return info

    def reload(self):
        """Hot-reload model artifacts from disk."""
        logger.info("🔄 Reloading model artifacts for %s...", self.model_name)
        self._load_artifacts()
        logger.info("   ✅ Model reloaded successfully")


def get_sample_transactions():
    """Return sample transactions for testing predictions."""
    legitimate = {
        "Time": 406.0,
        "V1": -0.24, "V2": 0.70, "V3": 0.53, "V4": 0.24,
        "V5": -0.28, "V6": -0.17, "V7": -0.05, "V8": 0.09,
        "V9": 0.23, "V10": -0.11, "V11": -0.63, "V12": -0.08,
        "V13": -0.31, "V14": 0.39, "V15": -0.45, "V16": 0.09,
        "V17": 0.27, "V18": -0.08, "V19": -0.09, "V20": -0.07,
        "V21": 0.10, "V22": 0.30, "V23": -0.01, "V24": 0.02,
        "V25": 0.31, "V26": 0.04, "V27": 0.04, "V28": 0.01,
        "Amount": 44.50,
    }

    fraud = {
        "Time": 80000.0,
        "V1": -3.04, "V2": -3.16, "V3": 1.09, "V4": 2.29,
        "V5": 1.36, "V6": -1.06, "V7": -2.74, "V8": 1.01,
        "V9": -0.68, "V10": -4.91, "V11": 3.93, "V12": -5.41,
        "V13": 0.14, "V14": -8.10, "V15": 0.30, "V16": -4.76,
        "V17": -7.43, "V18": -2.55, "V19": 0.69, "V20": 0.58,
        "V21": 0.53, "V22": 0.22, "V23": -0.15, "V24": -0.40,
        "V25": 0.15, "V26": -0.21, "V27": 0.73, "V28": 0.44,
        "Amount": 1.98,
    }

    return {"legitimate": legitimate, "fraud": fraud}
