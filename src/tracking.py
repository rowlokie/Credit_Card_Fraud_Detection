"""
MLflow Experiment Tracking Module
==================================
Wraps MLflow for experiment tracking, metric logging, model registration,
and artifact management.
"""

import hashlib
import numpy as np
import pandas as pd

from src.config import (
    MLFLOW_ENABLED, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    FIGURES_DIR
)
from src.logging_config import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """
    MLflow experiment tracking wrapper.

    Usage
    -----
    >>> tracker = MLflowTracker()
    >>> with tracker.start_run("training_v1"):
    ...     tracker.log_params({"n_estimators": 200})
    ...     tracker.log_metrics({"f1_score": 0.95})
    ...     tracker.log_model(model, "xgboost")
    """

    def __init__(self, experiment_name=None, tracking_uri=None):
        if not MLFLOW_ENABLED:
            logger.info("MLflow tracking is disabled")
            self.enabled = False
            return

        try:
            import mlflow
            import mlflow.sklearn
            import mlflow.xgboost
            self.mlflow = mlflow
            self.sklearn_module = mlflow.sklearn
            self.xgboost_module = mlflow.xgboost
            self.enabled = True
        except ImportError:
            logger.warning("MLflow not installed. Tracking disabled.")
            self.enabled = False
            return

        self.tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
        self.experiment_name = experiment_name or MLFLOW_EXPERIMENT_NAME
        self.active_run = None

        self.mlflow.set_tracking_uri(self.tracking_uri)
        self.mlflow.set_experiment(self.experiment_name)
        logger.info("📊 MLflow tracking initialized")
        logger.info("   URI: %s | Experiment: %s", self.tracking_uri, self.experiment_name)

    def start_run(self, run_name=None):
        """Start an MLflow run (context manager)."""
        if not self.enabled:
            return _NoOpContext()
        return self.mlflow.start_run(run_name=run_name)

    def log_params(self, params):
        """Log multiple parameters."""
        if not self.enabled:
            return
        # Flatten nested dicts and convert values to strings
        flat = {}
        for k, v in params.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat[f"{k}.{k2}"] = str(v2)
            else:
                flat[k] = str(v)
        self.mlflow.log_params(flat)

    def log_metrics(self, metrics, step=None):
        """Log multiple metrics."""
        if not self.enabled:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                self.mlflow.log_metric(key, float(value), step=step)

    def log_model(self, model, model_name):
        """Log a trained model as an MLflow artifact."""
        if not self.enabled:
            return

        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier

        if isinstance(model, XGBClassifier):
            self.xgboost_module.log_model(model, model_name)
        else:
            self.sklearn_module.log_model(model, model_name)
        logger.info("   📦 Model logged to MLflow: %s", model_name)

    def log_artifact(self, file_path):
        """Log a file as an artifact."""
        if not self.enabled:
            return
        self.mlflow.log_artifact(str(file_path))

    def log_artifacts_dir(self, dir_path):
        """Log all files in a directory as artifacts."""
        if not self.enabled:
            return
        self.mlflow.log_artifacts(str(dir_path))

    def log_figure(self, fig, filename):
        """Log a matplotlib figure as an artifact."""
        if not self.enabled:
            return
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            self.mlflow.log_artifact(path)

    def log_dataset_info(self, df, name="dataset"):
        """Log dataset metadata (hash, shape, class distribution)."""
        if not self.enabled:
            return

        # Dataset hash for versioning
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()[:12]

        self.log_params({
            f"{name}_hash": data_hash,
            f"{name}_rows": len(df),
            f"{name}_cols": len(df.columns),
        })

        if "Class" in df.columns:
            fraud_count = int(df["Class"].sum())
            self.log_params({
                f"{name}_fraud_count": fraud_count,
                f"{name}_fraud_pct": f"{fraud_count / len(df) * 100:.4f}%",
            })

    def log_cv_results(self, cv_results):
        """Log cross-validation results."""
        if not self.enabled:
            return
        self.log_metrics({
            "cv_mean": cv_results["mean"],
            "cv_std": cv_results["std"],
            "cv_folds": cv_results["cv_folds"],
        })
        for i, score in enumerate(cv_results["scores"]):
            self.log_metrics({f"cv_fold_{i}": score})

    def log_threshold(self, threshold_result):
        """Log threshold optimization results."""
        if not self.enabled:
            return
        self.log_metrics({
            "optimal_threshold": threshold_result["optimal_threshold"],
        })
        self.log_params({
            "threshold_strategy": threshold_result["strategy"],
        })
        for strategy, t in threshold_result.get("all_strategies", {}).items():
            self.log_metrics({f"threshold_{strategy}": t})

    def end_run(self):
        """End the active MLflow run."""
        if self.enabled:
            self.mlflow.end_run()


class _NoOpContext:
    """No-op context manager when MLflow is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
