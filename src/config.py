"""
Configuration Module
====================
Central configuration loaded from config.yaml with environment variable overrides.
All settings are accessible as module-level constants for backward compatibility.
"""

import os
from pathlib import Path

import yaml


# =============================================================================
# Project Root & Config Loading
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_yaml_config() -> dict:
    """Load configuration from YAML file."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _env(key: str, default=None, cast=None):
    """Get environment variable with optional type casting."""
    val = os.getenv(key, default)
    if val is None:
        return default
    if cast is not None and val is not default:
        return cast(val)
    return val


_cfg = _load_yaml_config()

# =============================================================================
# Directory Paths
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / (_cfg.get("data", {}).get("raw_path", "data/creditcard.csv").replace("data/", ""))

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# =============================================================================
# Pipeline Constants
# =============================================================================

RANDOM_STATE = int(_env("RANDOM_STATE", _cfg.get("training", {}).get("random_state", 42)))
TEST_SIZE = float(_cfg.get("data", {}).get("test_size", 0.2))
TARGET_COL = _cfg.get("data", {}).get("target_col", "Class")
SCALE_COLS = _cfg.get("data", {}).get("scale_cols", ["Time", "Amount"])
FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
N_PCA_COMPONENTS = None  # V1-V28 are already PCA-transformed

# =============================================================================
# Model Hyperparameters (defaults, overridden by optimizer when used)
# =============================================================================

_rf_cfg = _cfg.get("training", {}).get("models", {}).get("random_forest", {})
RANDOM_FOREST_PARAMS = {
    "n_estimators": _rf_cfg.get("n_estimators", 200),
    "max_depth": _rf_cfg.get("max_depth", 20),
    "min_samples_split": _rf_cfg.get("min_samples_split", 5),
    "min_samples_leaf": _rf_cfg.get("min_samples_leaf", 2),
    "max_features": _rf_cfg.get("max_features", "sqrt"),
    "class_weight": _rf_cfg.get("class_weight", "balanced"),
    "random_state": RANDOM_STATE,
    "n_jobs": _rf_cfg.get("n_jobs", -1),
    "verbose": 0,
}

_xgb_cfg = _cfg.get("training", {}).get("models", {}).get("xgboost", {})
XGBOOST_PARAMS = {
    "n_estimators": _xgb_cfg.get("n_estimators", 200),
    "max_depth": _xgb_cfg.get("max_depth", 8),
    "learning_rate": _xgb_cfg.get("learning_rate", 0.1),
    "subsample": _xgb_cfg.get("subsample", 0.8),
    "colsample_bytree": _xgb_cfg.get("colsample_bytree", 0.8),
    "reg_alpha": _xgb_cfg.get("reg_alpha", 0.1),
    "reg_lambda": _xgb_cfg.get("reg_lambda", 1.0),
    "random_state": RANDOM_STATE,
    "n_jobs": _xgb_cfg.get("n_jobs", -1),
    "eval_metric": _xgb_cfg.get("eval_metric", "logloss"),
    "verbosity": 0,
}

# =============================================================================
# SMOTE Configuration
# =============================================================================

_smote_cfg = _cfg.get("smote", {})
SMOTE_SAMPLING_STRATEGY = _smote_cfg.get("sampling_strategy", 0.5)
SMOTE_K_NEIGHBORS = _smote_cfg.get("k_neighbors", 5)

# =============================================================================
# Optimization Configuration
# =============================================================================

_opt_cfg = _cfg.get("optimization", {})
OPTIMIZATION_ENGINE = _env("OPTIMIZATION_ENGINE", _opt_cfg.get("engine", "optuna"))
OPTUNA_N_TRIALS = int(_env("OPTUNA_N_TRIALS", _opt_cfg.get("n_trials", 50)))
CV_FOLDS = int(_env("CV_FOLDS", _opt_cfg.get("cv_folds", 5)))
OPTIMIZATION_SCORING = _opt_cfg.get("scoring", "f1")
OPTIMIZATION_TIMEOUT = _opt_cfg.get("timeout", 600)

# =============================================================================
# Threshold Configuration
# =============================================================================

_thresh_cfg = _cfg.get("threshold", {})
THRESHOLD_STRATEGY = _env("THRESHOLD_STRATEGY", _thresh_cfg.get("strategy", "f1"))
COST_FP = float(_env("COST_FP", _thresh_cfg.get("cost_fp", 1.0)))
COST_FN = float(_env("COST_FN", _thresh_cfg.get("cost_fn", 10.0)))

# =============================================================================
# SHAP Configuration
# =============================================================================

_shap_cfg = _cfg.get("shap", {})
SHAP_ENABLED = _shap_cfg.get("enabled", True)
SHAP_BACKGROUND_SAMPLES = _shap_cfg.get("background_samples", 100)
SHAP_MAX_DISPLAY = _shap_cfg.get("max_display", 15)

# =============================================================================
# MLflow Configuration
# =============================================================================

_mlflow_cfg = _cfg.get("mlflow", {})
MLFLOW_ENABLED = _mlflow_cfg.get("enabled", True)
MLFLOW_TRACKING_URI = _env("MLFLOW_TRACKING_URI", _mlflow_cfg.get("tracking_uri", "mlruns"))
MLFLOW_EXPERIMENT_NAME = _env(
    "MLFLOW_EXPERIMENT_NAME",
    _mlflow_cfg.get("experiment_name", "credit-fraud-detection"),
)

# =============================================================================
# FastAPI Configuration
# =============================================================================

_api_cfg = _cfg.get("api", {})
API_HOST = _env("API_HOST", _api_cfg.get("host", "0.0.0.0"))
API_PORT = int(_env("API_PORT", _api_cfg.get("port", 8000)))
API_WORKERS = int(_env("API_WORKERS", _api_cfg.get("workers", 1)))
API_DEFAULT_MODEL = _env("DEFAULT_MODEL", _api_cfg.get("default_model", "xgboost"))
API_CORS_ORIGINS = _api_cfg.get("cors_origins", ["http://localhost:8501"])

# =============================================================================
# Frontend Configuration
# =============================================================================

_fe_cfg = _cfg.get("frontend", {})
FRONTEND_API_URL = _env("FRONTEND_API_URL", _fe_cfg.get("api_url", "http://localhost:8000"))
FRONTEND_FALLBACK = _fe_cfg.get("fallback_to_local", True)

# =============================================================================
# Visualization Settings
# =============================================================================

_viz_cfg = _cfg.get("visualization", {})
COLORS = _viz_cfg.get("colors", {
    "legitimate": "#2ecc71",
    "fraud": "#e74c3c",
    "primary": "#3498db",
    "secondary": "#9b59b6",
    "accent": "#f39c12",
    "background": "#2c3e50",
    "text": "#ecf0f1",
})
FIGURE_DPI = _viz_cfg.get("dpi", 150)
FIGURE_FORMAT = _viz_cfg.get("format", "png")
PLOT_STYLE = "seaborn-v0_8-darkgrid"

# =============================================================================
# Model File Paths
# =============================================================================


def get_model_path(model_name: str) -> Path:
    """Get the file path for saving/loading a model."""
    return MODELS_DIR / f"{model_name}.joblib"


def get_scaler_path() -> Path:
    """Get the file path for saving/loading the fitted scaler."""
    return MODELS_DIR / "scaler.joblib"


def get_metrics_path() -> Path:
    """Get the file path for saving training metrics."""
    return MODELS_DIR / "metrics.joblib"


def get_threshold_path(model_name: str) -> Path:
    """Get the file path for saving the optimal threshold."""
    return MODELS_DIR / f"{model_name}_threshold.joblib"


def get_shap_explainer_path(model_name: str) -> Path:
    """Get the file path for saving the SHAP explainer."""
    return MODELS_DIR / f"{model_name}_shap_explainer.joblib"
