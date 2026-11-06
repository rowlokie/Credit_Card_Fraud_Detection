"""
Visualization Module
====================
Professional-quality plots for EDA and model evaluation results.
All figures are saved to the outputs/figures/ directory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import COLORS, FIGURE_DPI, FIGURES_DIR, TARGET_COL
from src.logging_config import get_logger

logger = get_logger(__name__)


# Set global plot style
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560",
    "axes.labelcolor": "#eaeaea",
    "text.color": "#eaeaea",
    "xtick.color": "#eaeaea",
    "ytick.color": "#eaeaea",
    "grid.color": "#0f3460",
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})


# =============================================================================
# EDA VISUALIZATIONS
# =============================================================================

def plot_class_distribution(df, save=True):
    """Plot the class distribution (fraud vs. legitimate)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = df[TARGET_COL].value_counts()
    labels = ["Legitimate", "Fraudulent"]
    colors = [COLORS["legitimate"], COLORS["fraud"]]

    # Bar chart
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="white",
                       linewidth=1.5, width=0.6)
    for bar, count in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1000,
                     f'{count:,}', ha='center', va='bottom', fontweight='bold',
                     fontsize=12, color="#eaeaea")
    axes[0].set_title("Transaction Count by Class")
    axes[0].set_ylabel("Count")

    # Pie chart
    axes[1].pie(counts.values, labels=labels, colors=colors, autopct='%1.3f%%',
                startangle=90, explode=(0, 0.1), shadow=True,
                textprops={'fontsize': 12, 'color': '#eaeaea'})
    axes[1].set_title("Class Distribution")

    fig.suptitle("Class Imbalance in Credit Card Transactions",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=FIGURE_DPI,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_amount_distribution(df, save=True):
    """Plot transaction amount distributions for fraud vs. legitimate."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    legit = df[df[TARGET_COL] == 0]["Amount"]
    fraud = df[df[TARGET_COL] == 1]["Amount"]

    # Histogram
    axes[0].hist(legit, bins=80, alpha=0.7, color=COLORS["legitimate"],
                 label="Legitimate", density=True, edgecolor="none")
    axes[0].hist(fraud, bins=80, alpha=0.7, color=COLORS["fraud"],
                 label="Fraudulent", density=True, edgecolor="none")
    axes[0].set_title("Transaction Amount Distribution")
    axes[0].set_xlabel("Amount ($)")
    axes[0].set_ylabel("Density")
    axes[0].set_xlim(0, 500)
    axes[0].legend()

    # Box plot
    box_data = [legit, fraud]
    bp = axes[1].boxplot(box_data, tick_labels=["Legitimate", "Fraudulent"],
                         patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], [COLORS["legitimate"], COLORS["fraud"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color='#eaeaea')
    axes[1].set_title("Amount Box Plot (outliers hidden)")
    axes[1].set_ylabel("Amount ($)")

    fig.suptitle("Transaction Amounts: Fraud vs Legitimate",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "amount_distribution.png", dpi=FIGURE_DPI,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_time_distribution(df, save=True):
    """Plot transaction time distributions."""
    fig, ax = plt.subplots(figsize=(14, 5))

    legit = df[df[TARGET_COL] == 0]["Time"] / 3600
    fraud = df[df[TARGET_COL] == 1]["Time"] / 3600

    ax.hist(legit, bins=48, alpha=0.6, color=COLORS["legitimate"],
            label="Legitimate", density=True, edgecolor="none")
    ax.hist(fraud, bins=48, alpha=0.7, color=COLORS["fraud"],
            label="Fraudulent", density=True, edgecolor="none")
    ax.set_title("Transaction Time Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("Time (hours since first transaction)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=12)

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "time_distribution.png", dpi=FIGURE_DPI,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_correlation_heatmap(df, save=True):
    """Plot correlation heatmap of features."""
    fig, ax = plt.subplots(figsize=(16, 12))

    # Select a subset of features for readability
    cols_to_plot = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V10",
                    "V11", "V12", "V14", "V16", "V17", "Amount", "Class"]
    corr = df[cols_to_plot].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True,
                fmt=".2f", square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Correlation"},
                ax=ax, annot_kws={"size": 8})

    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=FIGURE_DPI,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_fraud_features(df, save=True):
    """Plot distributions of top features that differ between fraud and legitimate."""
    top_features = ["V17", "V14", "V12", "V10", "V16", "V3", "V7", "V11"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, feature in enumerate(top_features):
        legit = df[df[TARGET_COL] == 0][feature]
        fraud = df[df[TARGET_COL] == 1][feature]

        axes[i].hist(legit, bins=50, alpha=0.6, color=COLORS["legitimate"],
                     label="Legit", density=True, edgecolor="none")
        axes[i].hist(fraud, bins=50, alpha=0.7, color=COLORS["fraud"],
                     label="Fraud", density=True, edgecolor="none")
        axes[i].set_title(feature, fontsize=13, fontweight="bold")
        axes[i].legend(fontsize=8)

    fig.suptitle("Feature Distributions: Fraud vs Legitimate (Top PCA Components)",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "fraud_features.png", dpi=FIGURE_DPI,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def run_eda(df):
    """Run all EDA visualizations."""
    logger.info("🎨 GENERATING EDA VISUALIZATIONS")
    logger.info("─" * 40)

    plot_class_distribution(df)
    logger.info("   ✅ Class distribution plot saved")

    plot_amount_distribution(df)
    logger.info("   ✅ Amount distribution plot saved")

    plot_time_distribution(df)
    logger.info("   ✅ Time distribution plot saved")

    plot_correlation_heatmap(df)
    logger.info("   ✅ Correlation heatmap saved")

    plot_fraud_features(df)
    logger.info("   ✅ Fraud feature distributions saved")

    logger.info("   📁 All figures saved to: %s", FIGURES_DIR)
    plt.close("all")


# =============================================================================
# MODEL EVALUATION VISUALIZATIONS
# =============================================================================

def plot_confusion_matrices(all_metrics, save=True):
    """Plot confusion matrices for all models side by side."""
    n_models = len(all_metrics)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    for ax, metrics in zip(axes, all_metrics):
        cm = metrics["confusion_matrix"]
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                    xticklabels=["Legitimate", "Fraud"],
                    yticklabels=["Legitimate", "Fraud"],
                    linewidths=2, linecolor="#1a1a2e",
                    cbar_kws={"shrink": 0.8},
                    annot_kws={"size": 14, "fontweight": "bold"})

        # Add percentages as secondary annotations
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.75,
                        f"({cm_normalized[i, j]:.1%})",
                        ha="center", va="center",
                        fontsize=10, color="#eaeaea", alpha=0.8)

        ax.set_title(f"{metrics['model_name']}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=FIGURE_DPI,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_roc_curves(all_metrics, save=True):
    """Plot ROC curves for all models on the same axes."""
    fig, ax = plt.subplots(figsize=(10, 8))

    model_colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]

    for i, metrics in enumerate(all_metrics):
        roc_data = metrics["roc_curve"]
        color = model_colors[i % len(model_colors)]
        ax.plot(roc_data["fpr"], roc_data["tpr"],
                color=color, linewidth=2.5,
                label=f'{metrics["model_name"]} (AUC = {metrics["auc_roc"]:.4f})')

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'w--', alpha=0.4, linewidth=1.5, label="Random Classifier")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves — Model Comparison", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12, framealpha=0.3)

    # Fill area under best curve
    best_idx = np.argmax([m["auc_roc"] for m in all_metrics])
    best_roc = all_metrics[best_idx]["roc_curve"]
    ax.fill_between(best_roc["fpr"], best_roc["tpr"],
                    alpha=0.1, color=model_colors[best_idx])

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=FIGURE_DPI,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_precision_recall_curves(all_metrics, save=True):
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))

    model_colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]

    for i, metrics in enumerate(all_metrics):
        pr_data = metrics["pr_curve"]
        color = model_colors[i % len(model_colors)]
        ax.plot(pr_data["recall"], pr_data["precision"],
                color=color, linewidth=2.5,
                label=f'{metrics["model_name"]} (AUC-PR = {metrics["auc_pr"]:.4f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curves — Model Comparison",
                 fontsize=16, fontweight="bold")
    ax.legend(loc="lower left", fontsize=12, framealpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "precision_recall_curves.png", dpi=FIGURE_DPI,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_feature_importance(model, feature_names, model_name="Model", top_n=15, save=True):
    """Plot top feature importances."""
    fig, ax = plt.subplots(figsize=(10, 8))

    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, top_n))

    ax.barh(range(top_n), importances[indices], color=colors,
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(top_n))

    feature_names_arr = np.array(feature_names) if not isinstance(feature_names, np.ndarray) else feature_names
    ax.set_yticklabels(feature_names_arr[indices])
    ax.set_xlabel("Importance", fontsize=13)
    ax.set_title(f"{model_name} — Top {top_n} Feature Importances",
                 fontsize=16, fontweight="bold")

    plt.tight_layout()
    if save:
        safe_name = model_name.lower().replace(" ", "_")
        fig.savefig(FIGURES_DIR / f"feature_importance_{safe_name}.png",
                    dpi=FIGURE_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_metrics_comparison(all_metrics, save=True):
    """Plot a bar chart comparing key metrics across models."""
    fig, ax = plt.subplots(figsize=(12, 7))

    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    model_colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]

    x = np.arange(len(metric_names))
    width = 0.35
    n_models = len(all_metrics)

    for i, metrics in enumerate(all_metrics):
        offset = (i - (n_models - 1) / 2) * width
        values = [metrics[k] for k in metric_keys]
        bars = ax.bar(x + offset, values, width * 0.9,
                      label=metrics["model_name"],
                      color=model_colors[i % len(model_colors)],
                      edgecolor="white", linewidth=0.5)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color="#eaeaea")

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Model Performance Comparison", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, framealpha=0.3)
    ax.axhline(y=0.9, color="#e94560", linestyle="--", alpha=0.3, label="0.9 baseline")

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "metrics_comparison.png", dpi=FIGURE_DPI,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def plot_all_results(all_metrics, models, feature_names):
    """Generate all evaluation visualizations."""
    logger.info("🎨 GENERATING EVALUATION VISUALIZATIONS")
    logger.info("─" * 40)

    plot_confusion_matrices(all_metrics)
    logger.info("   ✅ Confusion matrices saved")

    plot_roc_curves(all_metrics)
    logger.info("   ✅ ROC curves saved")

    plot_precision_recall_curves(all_metrics)
    logger.info("   ✅ Precision-Recall curves saved")

    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            plot_feature_importance(model, feature_names, model_name=name)
            logger.info("   ✅ Feature importance (%s) saved", name)

    plot_metrics_comparison(all_metrics)
    logger.info("   ✅ Metrics comparison chart saved")

    logger.info("   📁 All figures saved to: %s", FIGURES_DIR)
    plt.close("all")
