"""
SHAP Explainability Module
===========================
Provides SHAP-based model explanations for both Random Forest and XGBoost.
Generates summary plots, force plots, waterfall plots, and per-prediction
explanations for API and Streamlit consumption.
"""

import numpy as np
import pandas as pd
import joblib

from src.config import (
    SHAP_BACKGROUND_SAMPLES, SHAP_MAX_DISPLAY,
    FEATURE_COLS, FIGURES_DIR, FIGURE_DPI, get_shap_explainer_path
)
from src.logging_config import get_logger

logger = get_logger(__name__)


class FraudExplainer:
    """
    SHAP-based explainer for fraud detection models.

    Parameters
    ----------
    model : estimator
        Trained model (RandomForest or XGBoost).
    model_name : str
        Name of the model for file paths.
    background_data : pd.DataFrame or np.ndarray, optional
        Background data for TreeExplainer. If None, uses a sample.
    """

    def __init__(self, model, model_name="model", background_data=None):
        try:
            import shap
            self.shap = shap
        except ImportError:
            logger.error("SHAP not installed. Install with: pip install shap")
            raise

        self.model = model
        self.model_name = model_name
        self.explainer = None
        self.background_data = background_data
        self._init_explainer()

    def _init_explainer(self):
        """Initialize the SHAP TreeExplainer."""
        logger.info("🔍 Initializing SHAP TreeExplainer for %s...", self.model_name)
        try:
            self.explainer = self.shap.TreeExplainer(self.model)
            logger.info("   ✅ SHAP TreeExplainer ready")
        except Exception as e:
            logger.warning("TreeExplainer failed, falling back to KernelExplainer: %s", e)
            if self.background_data is not None:
                bg = self.shap.sample(self.background_data, SHAP_BACKGROUND_SAMPLES)
                self.explainer = self.shap.KernelExplainer(self.model.predict_proba, bg)
            else:
                raise

    def compute_shap_values(self, X, max_samples=None):
        """
        Compute SHAP values for given data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature data.
        max_samples : int, optional
            Maximum samples to explain (for performance).

        Returns
        -------
        np.ndarray
            SHAP values array.
        """
        if max_samples and len(X) > max_samples:
            if isinstance(X, pd.DataFrame):
                X = X.sample(max_samples, random_state=42)
            else:
                idx = np.random.choice(len(X), max_samples, replace=False)
                X = X[idx]

        logger.info("   Computing SHAP values for %d samples...", len(X))
        shap_values = self.explainer.shap_values(X)

        # For binary classification, shap_values may be a list [class_0, class_1]
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # Use fraud class

        logger.info("   ✅ SHAP values computed. Shape: %s", shap_values.shape)
        return shap_values

    def explain_single(self, transaction_features, feature_names=None):
        """
        Explain a single prediction with SHAP values.

        Parameters
        ----------
        transaction_features : dict or pd.DataFrame or np.ndarray
            Single transaction features.
        feature_names : list, optional
            Feature names.

        Returns
        -------
        dict
            Explanation with top contributing features and SHAP values.
        """
        feature_names = feature_names or FEATURE_COLS

        if isinstance(transaction_features, dict):
            X = pd.DataFrame([transaction_features])[feature_names]
        elif isinstance(transaction_features, pd.DataFrame):
            X = transaction_features[feature_names]
        else:
            X = np.array(transaction_features).reshape(1, -1)

        shap_values = self.explainer.shap_values(X)

        # Handle different SHAP output formats:
        # - List of 2 arrays [class_0, class_1]: each shape (n_samples, n_features)
        # - 3D array (n_samples, n_features, n_classes): e.g. from RandomForest
        # - 2D array (n_samples, n_features): single output
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_vals = np.array(shap_values[1][0])  # Fraud class, first sample
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_vals = shap_values[0, :, 1]  # First sample, all features, fraud class
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            shap_vals = shap_values[0]
        else:
            shap_vals = np.array(shap_values[0])

        # Ensure shap_vals is a 1D array of scalars
        shap_vals = np.array(shap_vals).flatten()

        # Build explanation
        feature_contributions = []
        for i, (name, val) in enumerate(zip(feature_names, shap_vals)):
            sv = float(np.asarray(val).item())
            fv = float(X.iloc[0, i]) if isinstance(X, pd.DataFrame) else float(X[0, i])
            feature_contributions.append({
                "feature": name,
                "shap_value": sv,
                "abs_shap_value": abs(sv),
                "feature_value": fv,
            })

        # Sort by absolute SHAP value
        feature_contributions.sort(key=lambda x: x["abs_shap_value"], reverse=True)

        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(np.asarray(base_value).flatten()[1]) if len(base_value) == 2 else float(np.asarray(base_value).flatten()[0])

        return {
            "base_value": float(base_value),
            "shap_values": shap_vals.tolist(),
            "top_features": feature_contributions[:SHAP_MAX_DISPLAY],
            "all_features": feature_contributions,
        }

    def plot_summary(self, X, feature_names=None, save=True):
        """Generate and save SHAP summary (beeswarm) plot."""
        import matplotlib.pyplot as plt

        feature_names = feature_names or FEATURE_COLS
        shap_values = self.compute_shap_values(X)

        fig, ax = plt.subplots(figsize=(12, 8))
        self.shap.summary_plot(
            shap_values, X, feature_names=feature_names,
            max_display=SHAP_MAX_DISPLAY, show=False, plot_type="dot"
        )
        plt.title(f"SHAP Summary — {self.model_name}", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save:
            safe_name = self.model_name.lower().replace(" ", "_")
            path = FIGURES_DIR / f"shap_summary_{safe_name}.png"
            plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
            logger.info("   ✅ SHAP summary plot saved: %s", path)

        plt.close()
        return fig

    def plot_bar_importance(self, X, feature_names=None, save=True):
        """Generate SHAP bar importance plot."""
        import matplotlib.pyplot as plt

        feature_names = feature_names or FEATURE_COLS
        shap_values = self.compute_shap_values(X)

        fig, ax = plt.subplots(figsize=(12, 8))
        self.shap.summary_plot(
            shap_values, X, feature_names=feature_names,
            max_display=SHAP_MAX_DISPLAY, show=False, plot_type="bar"
        )
        plt.title(f"SHAP Feature Importance — {self.model_name}",
                  fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save:
            safe_name = self.model_name.lower().replace(" ", "_")
            path = FIGURES_DIR / f"shap_importance_{safe_name}.png"
            plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
            logger.info("   ✅ SHAP importance plot saved: %s", path)

        plt.close()
        return fig

    def plot_waterfall(self, X_single, feature_names=None, save=True):
        """Generate SHAP waterfall plot for a single prediction."""
        import matplotlib.pyplot as plt

        feature_names = feature_names or FEATURE_COLS
        explanation = self.explain_single(X_single, feature_names)

        base_value = explanation["base_value"]
        shap_vals = np.array(explanation["shap_values"])

        if isinstance(X_single, dict):
            X_arr = pd.DataFrame([X_single])[feature_names]
        elif isinstance(X_single, pd.DataFrame):
            X_arr = X_single[feature_names]
        else:
            X_arr = np.array(X_single).reshape(1, -1)

        shap_explanation = self.shap.Explanation(
            values=shap_vals,
            base_values=base_value,
            data=X_arr.values[0] if isinstance(X_arr, pd.DataFrame) else X_arr[0],
            feature_names=feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        self.shap.plots.waterfall(shap_explanation, max_display=SHAP_MAX_DISPLAY, show=False)
        plt.title(f"SHAP Waterfall — {self.model_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            safe_name = self.model_name.lower().replace(" ", "_")
            path = FIGURES_DIR / f"shap_waterfall_{safe_name}.png"
            plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
            logger.info("   ✅ SHAP waterfall plot saved: %s", path)

        plt.close()
        return fig

    def save_explainer(self):
        """Save the SHAP explainer to disk."""
        path = get_shap_explainer_path(self.model_name.lower().replace(" ", "_"))
        joblib.dump(self.explainer, path)
        logger.info("   💾 SHAP explainer saved: %s", path)

    @staticmethod
    def load_explainer(model_name):
        """Load a saved SHAP explainer from disk."""
        path = get_shap_explainer_path(model_name)
        if not path.exists():
            return None
        return joblib.load(path)
