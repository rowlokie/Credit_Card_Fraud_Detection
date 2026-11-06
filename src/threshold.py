"""
Threshold Optimization Module
==============================
Finds the optimal decision threshold for fraud detection using multiple
strategies: F1 maximization, Youden's J statistic, cost-sensitive, and
precision-recall break-even.
"""

import numpy as np
import joblib
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve
)

from src.config import (
    THRESHOLD_STRATEGY, COST_FP, COST_FN,
    get_threshold_path, FIGURES_DIR, FIGURE_DPI
)
from src.logging_config import get_logger

logger = get_logger(__name__)


class ThresholdOptimizer:
    """
    Finds and applies optimal classification thresholds.

    Parameters
    ----------
    strategy : str
        Optimization strategy: "f1", "youden", "cost", "precision_recall"
    cost_fp : float
        Cost of a false positive.
    cost_fn : float
        Cost of a false negative (missed fraud).
    """

    def __init__(self, strategy=None, cost_fp=None, cost_fn=None):
        self.strategy = strategy or THRESHOLD_STRATEGY
        self.cost_fp = cost_fp or COST_FP
        self.cost_fn = cost_fn or COST_FN
        self.optimal_threshold = 0.5
        self.all_results = {}

    def find_optimal(self, y_true, y_proba, search_range=(0.01, 0.99), n_steps=1000):
        """
        Find the optimal threshold using the configured strategy.

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_proba : array-like
            Predicted probabilities for the positive class.
        search_range : tuple
            (min_threshold, max_threshold) to search.
        n_steps : int
            Number of threshold values to evaluate.

        Returns
        -------
        dict
            Optimization results including optimal threshold and metrics.
        """
        logger.info("🎯 THRESHOLD OPTIMIZATION (strategy=%s)", self.strategy)
        logger.info("─" * 40)

        thresholds = np.linspace(search_range[0], search_range[1], n_steps)

        # Compute all strategies
        self.all_results = {
            "f1": self._optimize_f1(y_true, y_proba, thresholds),
            "youden": self._optimize_youden(y_true, y_proba),
            "cost": self._optimize_cost(y_true, y_proba, thresholds),
            "precision_recall": self._optimize_pr_breakeven(y_true, y_proba),
        }

        # Use selected strategy
        result = self.all_results[self.strategy]
        self.optimal_threshold = result["threshold"]

        logger.info("   Strategy Results:")
        for name, res in self.all_results.items():
            marker = " ← SELECTED" if name == self.strategy else ""
            logger.info("     %s: threshold=%.4f%s", name, res["threshold"], marker)

        logger.info("   ✅ Optimal threshold: %.4f (strategy=%s)",
                     self.optimal_threshold, self.strategy)

        return {
            "optimal_threshold": self.optimal_threshold,
            "strategy": self.strategy,
            "all_strategies": {k: v["threshold"] for k, v in self.all_results.items()},
            **result,
        }

    def _optimize_f1(self, y_true, y_proba, thresholds):
        """Find threshold that maximizes F1-score."""
        f1_scores = []
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

        best_idx = np.argmax(f1_scores)
        return {
            "threshold": float(thresholds[best_idx]),
            "f1_score": float(f1_scores[best_idx]),
            "thresholds": thresholds,
            "scores": np.array(f1_scores),
        }

    def _optimize_youden(self, y_true, y_proba):
        """Find threshold using Youden's J statistic (ROC-based)."""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return {
            "threshold": float(thresholds[best_idx]),
            "j_score": float(j_scores[best_idx]),
            "tpr": float(tpr[best_idx]),
            "fpr": float(fpr[best_idx]),
        }

    def _optimize_cost(self, y_true, y_proba, thresholds):
        """Find threshold that minimizes total cost (cost-sensitive)."""
        costs = []
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            total_cost = fp * self.cost_fp + fn * self.cost_fn
            costs.append(total_cost)

        best_idx = np.argmin(costs)
        return {
            "threshold": float(thresholds[best_idx]),
            "min_cost": float(costs[best_idx]),
            "cost_fp": self.cost_fp,
            "cost_fn": self.cost_fn,
        }

    def _optimize_pr_breakeven(self, y_true, y_proba):
        """Find threshold at precision-recall break-even point."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        # Find where precision ≈ recall
        diff = np.abs(precision[:-1] - recall[:-1])
        best_idx = np.argmin(diff)
        return {
            "threshold": float(thresholds[best_idx]),
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
        }

    def apply_threshold(self, y_proba, threshold=None):
        """Apply threshold to probabilities to get predictions."""
        t = threshold or self.optimal_threshold
        return (y_proba >= t).astype(int)

    def save(self, model_name):
        """Save optimal threshold to disk."""
        path = get_threshold_path(model_name)
        data = {
            "threshold": self.optimal_threshold,
            "strategy": self.strategy,
            "all_results": {k: v.get("threshold") for k, v in self.all_results.items()},
        }
        joblib.dump(data, path)
        logger.info("   💾 Threshold saved: %s", path)

    @staticmethod
    def load(model_name):
        """Load saved threshold from disk."""
        path = get_threshold_path(model_name)
        if not path.exists():
            return None
        return joblib.load(path)

    def plot_threshold_curve(self, save=True):
        """Plot F1 score vs threshold curve."""
        import matplotlib.pyplot as plt

        f1_result = self.all_results.get("f1")
        if f1_result is None or "thresholds" not in f1_result:
            logger.warning("No F1 threshold data available for plotting")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(f1_result["thresholds"], f1_result["scores"],
                color="#3498db", linewidth=2, label="F1-Score")
        ax.axvline(x=self.optimal_threshold, color="#e74c3c",
                   linestyle="--", linewidth=1.5,
                   label=f"Optimal ({self.optimal_threshold:.3f})")
        ax.set_xlabel("Decision Threshold", fontsize=13)
        ax.set_ylabel("F1-Score", fontsize=13)
        ax.set_title("F1-Score vs Decision Threshold", fontsize=16, fontweight="bold")
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            fig.savefig(FIGURES_DIR / "threshold_optimization.png",
                        dpi=FIGURE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("   ✅ Threshold plot saved")
        return fig
