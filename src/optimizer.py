"""
Hyperparameter Optimization Module
====================================
Supports Optuna (Bayesian) and RandomizedSearchCV for hyperparameter tuning
with Stratified K-Fold cross-validation.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

from src.config import (
    RANDOM_STATE, OPTUNA_N_TRIALS, CV_FOLDS,
    OPTIMIZATION_SCORING, OPTIMIZATION_TIMEOUT, OPTIMIZATION_ENGINE
)
from src.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Optuna Objective Functions
# =============================================================================

def _rf_objective(trial, X, y, cv_folds, scoring):
    """Optuna objective function for Random Forest."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    model = RandomForestClassifier(**params)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    return scores.mean()


def _xgb_objective(trial, X, y, cv_folds, scoring, scale_pos_weight=None):
    """Optuna objective function for XGBoost."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "verbosity": 0,
    }

    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight

    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    return scores.mean()


# =============================================================================
# Optuna Optimization
# =============================================================================

def optimize_with_optuna(X, y, model_type="xgboost", n_trials=None,
                         cv_folds=None, scoring=None, timeout=None,
                         scale_pos_weight=None):
    """
    Run Optuna hyperparameter optimization.

    Parameters
    ----------
    X : array-like
        Training features.
    y : array-like
        Training labels.
    model_type : str
        "random_forest" or "xgboost"
    n_trials : int, optional
        Number of Optuna trials.
    cv_folds : int, optional
        Number of CV folds.
    scoring : str, optional
        Scoring metric (e.g., "f1", "roc_auc").
    timeout : int, optional
        Max seconds for optimization.
    scale_pos_weight : float, optional
        Class weight ratio for XGBoost.

    Returns
    -------
    dict
        Best hyperparameters found.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        logger.error("Optuna not installed. Install with: pip install optuna")
        raise

    n_trials = n_trials or OPTUNA_N_TRIALS
    cv_folds = cv_folds or CV_FOLDS
    scoring = scoring or OPTIMIZATION_SCORING
    timeout = timeout or OPTIMIZATION_TIMEOUT

    logger.info("🔬 OPTUNA OPTIMIZATION — %s", model_type.upper())
    logger.info("   Trials: %d | CV Folds: %d | Scoring: %s | Timeout: %ds",
                n_trials, cv_folds, scoring, timeout)

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    if model_type == "random_forest":
        study.optimize(
            lambda trial: _rf_objective(trial, X, y, cv_folds, scoring),
            n_trials=n_trials, timeout=timeout, show_progress_bar=True
        )
    elif model_type == "xgboost":
        study.optimize(
            lambda trial: _xgb_objective(trial, X, y, cv_folds, scoring, scale_pos_weight),
            n_trials=n_trials, timeout=timeout, show_progress_bar=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    best_params = study.best_params
    best_score = study.best_value

    logger.info("   ✅ Best %s score: %.4f", scoring, best_score)
    logger.info("   Best params: %s", best_params)
    logger.info("   Completed trials: %d", len(study.trials))

    return {
        "best_params": best_params,
        "best_score": best_score,
        "n_trials": len(study.trials),
        "study": study,
    }


# =============================================================================
# RandomizedSearchCV Fallback
# =============================================================================

# Parameter distributions for RandomizedSearchCV
RF_PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(100, 500),
    "max_depth": randint(5, 50),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced", "balanced_subsample"],
}

XGB_PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(100, 500),
    "max_depth": randint(3, 15),
    "learning_rate": uniform(0.01, 0.29),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "reg_alpha": uniform(0.001, 9.999),
    "reg_lambda": uniform(0.001, 9.999),
    "min_child_weight": randint(1, 10),
}


def optimize_with_randomized_search(X, y, model_type="xgboost", n_iter=100,
                                     cv_folds=None, scoring=None,
                                     scale_pos_weight=None):
    """
    Run RandomizedSearchCV hyperparameter optimization.

    Returns
    -------
    dict
        Best hyperparameters found.
    """
    cv_folds = cv_folds or CV_FOLDS
    scoring = scoring or OPTIMIZATION_SCORING

    logger.info("🔬 RANDOMIZED SEARCH CV — %s", model_type.upper())
    logger.info("   Iterations: %d | CV Folds: %d | Scoring: %s", n_iter, cv_folds, scoring)

    if model_type == "random_forest":
        base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        param_dist = RF_PARAM_DISTRIBUTIONS
    elif model_type == "xgboost":
        xgb_params = {"random_state": RANDOM_STATE, "n_jobs": -1,
                       "eval_metric": "logloss", "verbosity": 0}
        if scale_pos_weight is not None:
            xgb_params["scale_pos_weight"] = scale_pos_weight
        base_model = XGBClassifier(**xgb_params)
        param_dist = XGB_PARAM_DISTRIBUTIONS
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base_model, param_distributions=param_dist,
        n_iter=n_iter, cv=cv, scoring=scoring,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    search.fit(X, y)

    logger.info("   ✅ Best %s score: %.4f", scoring, search.best_score_)
    logger.info("   Best params: %s", search.best_params_)

    return {
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "cv_results": search.cv_results_,
        "search": search,
    }


# =============================================================================
# Cross-Validation Scoring
# =============================================================================

def cross_validate_model(model, X, y, cv_folds=None, scoring=None):
    """
    Run stratified cross-validation and return scores.

    Parameters
    ----------
    model : estimator
        Model to evaluate.
    X : array-like
        Features.
    y : array-like
        Labels.
    cv_folds : int, optional
        Number of CV folds.
    scoring : str or list, optional
        Scoring metric(s).

    Returns
    -------
    dict
        CV scores with mean and std.
    """
    cv_folds = cv_folds or CV_FOLDS
    scoring = scoring or OPTIMIZATION_SCORING

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    logger.info("📊 Cross-Validation (%d-fold, scoring=%s)", cv_folds, scoring)

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    logger.info("   Scores: %s", [f"{s:.4f}" for s in scores])
    logger.info("   Mean: %.4f (±%.4f)", scores.mean(), scores.std())

    return {
        "scores": scores.tolist(),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "cv_folds": cv_folds,
        "scoring": scoring,
    }


# =============================================================================
# Unified Optimizer Entry Point
# =============================================================================

def run_optimization(X, y, model_type="xgboost", engine=None, **kwargs):
    """
    Run hyperparameter optimization using the configured engine.

    Parameters
    ----------
    X : array-like
        Training features.
    y : array-like
        Training labels.
    model_type : str
        "random_forest" or "xgboost"
    engine : str, optional
        "optuna" or "randomized_search". Defaults to config setting.

    Returns
    -------
    dict
        Optimization results with best_params and best_score.
    """
    engine = engine or OPTIMIZATION_ENGINE

    if engine == "optuna":
        return optimize_with_optuna(X, y, model_type=model_type, **kwargs)
    elif engine == "randomized_search":
        return optimize_with_randomized_search(X, y, model_type=model_type, **kwargs)
    else:
        raise ValueError(f"Unknown optimization engine: {engine}")
