"""
Evaluation Module
=================
Comprehensive model evaluation with multiple metrics relevant to
fraud detection. Supports threshold-optimized evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

from src.logging_config import get_logger

logger = get_logger(__name__)


def evaluate_model(model, X_test, y_test, model_name="Model", threshold=None):
    """
    Evaluate a trained model with comprehensive metrics.

    Parameters
    ----------
    model : estimator
        Fitted model with predict and predict_proba methods.
    X_test : array-like
        Test features.
    y_test : array-like
        True test labels.
    model_name : str
        Name for display purposes.
    threshold : float, optional
        Custom decision threshold. If None, uses default 0.5.

    Returns
    -------
    dict
        Dictionary containing all computed metrics.
    """
    y_proba = model.predict_proba(X_test)[:, 1]

    if threshold is not None:
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        threshold = 0.5

    metrics = {
        "model_name": model_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "auc_pr": average_precision_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm
    metrics["true_negatives"] = cm[0, 0]
    metrics["false_positives"] = cm[0, 1]
    metrics["false_negatives"] = cm[1, 0]
    metrics["true_positives"] = cm[1, 1]

    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
    metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds}

    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
    metrics["pr_curve"] = {
        "precision": pr_precision,
        "recall": pr_recall,
        "thresholds": pr_thresholds,
    }

    metrics["classification_report"] = classification_report(
        y_test, y_pred, target_names=["Legitimate", "Fraud"]
    )

    metrics["y_pred"] = y_pred
    metrics["y_proba"] = y_proba

    return metrics


def print_metrics(metrics):
    """Print evaluation metrics in a formatted table."""
    name = metrics["model_name"]
    threshold = metrics.get("threshold", 0.5)
    logger.info("=" * 60)
    logger.info("  📊 %s — EVALUATION RESULTS (threshold=%.3f)", name, threshold)
    logger.info("=" * 60)
    logger.info("  Accuracy:     %.4f  (%.2f%%)", metrics["accuracy"], metrics["accuracy"] * 100)
    logger.info("  Precision:    %.4f  (%.2f%%)", metrics["precision"], metrics["precision"] * 100)
    logger.info("  Recall:       %.4f  (%.2f%%)", metrics["recall"], metrics["recall"] * 100)
    logger.info("  F1-Score:     %.4f  (%.2f%%)", metrics["f1_score"], metrics["f1_score"] * 100)
    logger.info("  AUC-ROC:      %.4f", metrics["auc_roc"])
    logger.info("  AUC-PR:       %.4f", metrics["auc_pr"])
    logger.info("  ─" * 20)
    logger.info("  Confusion Matrix:")
    logger.info("    True Negatives:  %6s", f"{metrics['true_negatives']:,}")
    logger.info("    False Positives: %6s", f"{metrics['false_positives']:,}")
    logger.info("    False Negatives: %6s", f"{metrics['false_negatives']:,}")
    logger.info("    True Positives:  %6s", f"{metrics['true_positives']:,}")
    logger.info("=" * 60)


def compare_models(all_metrics):
    """Print a side-by-side comparison table of all models."""
    comparison_data = []
    for m in all_metrics:
        comparison_data.append({
            "Model": m["model_name"],
            "Threshold": f"{m.get('threshold', 0.5):.3f}",
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1-Score": f"{m['f1_score']:.4f}",
            "AUC-ROC": f"{m['auc_roc']:.4f}",
            "AUC-PR": f"{m['auc_pr']:.4f}",
        })

    df = pd.DataFrame(comparison_data)
    logger.info("=" * 80)
    logger.info("  📊 MODEL COMPARISON")
    logger.info("=" * 80)
    logger.info("\n%s", df.to_string(index=False))
    logger.info("=" * 80)

    return df


def get_best_model(all_metrics, metric="f1_score"):
    """Determine the best model based on a specified metric."""
    best = max(all_metrics, key=lambda m: m[metric])
    logger.info("   🏆 Best model by %s: %s (%s=%.4f)",
                metric, best["model_name"], metric, best[metric])
    return best
