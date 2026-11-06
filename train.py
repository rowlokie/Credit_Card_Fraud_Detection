"""
Training Pipeline
=================
End-to-end training script with cross-validation, Optuna hyperparameter
optimization, threshold optimization, SHAP explainability, and MLflow tracking.

Usage:
    python train.py                          # Standard training
    python train.py --optimize               # With hyperparameter optimization
    python train.py --optimize --n-trials 30 # Custom Optuna trials
    python train.py --no-eda --no-smote      # Skip EDA and SMOTE
"""

import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

from src.config import FEATURE_COLS, FIGURES_DIR, MODELS_DIR, SHAP_ENABLED, MLFLOW_ENABLED
from src.data_loader import load_dataset, get_dataset_summary
from src.preprocessing import preprocess_pipeline
from src.balancing import apply_smote
from src.models import (
    get_all_models, train_model, save_model,
    create_model_from_params
)
from src.evaluation import evaluate_model, print_metrics, compare_models, get_best_model
from src.visualizations import run_eda, plot_all_results
from src.threshold import ThresholdOptimizer
from src.logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train credit card fraud detection models."
    )
    parser.add_argument("--no-eda", action="store_true", help="Skip EDA visualizations.")
    parser.add_argument("--no-smote", action="store_true", help="Skip SMOTE class balancing.")
    parser.add_argument("--no-plots", action="store_true", help="Skip all plot generation.")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization.")
    parser.add_argument("--engine", type=str, default=None,
                        choices=["optuna", "randomized_search"],
                        help="Optimization engine (default: config setting).")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Number of Optuna trials (default: 50).")
    parser.add_argument("--cv-folds", type=int, default=None,
                        help="Cross-validation folds (default: 5).")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP analysis.")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow tracking.")
    parser.add_argument("--no-threshold", action="store_true", help="Skip threshold optimization.")
    return parser.parse_args()


def main():
    """Run the full training pipeline."""
    args = parse_args()
    start_time = time.time()

    logger.info("═" * 60)
    logger.info("  🔍 CREDIT CARD FRAUD DETECTION — TRAINING PIPELINE")
    logger.info("═" * 60)

    # =========================================================================
    # MLflow Tracking Setup
    # =========================================================================
    tracker = None
    if MLFLOW_ENABLED and not args.no_mlflow:
        try:
            from src.tracking import MLflowTracker
            tracker = MLflowTracker()
        except Exception as e:
            logger.warning("MLflow initialization failed: %s", e)

    mlflow_context = tracker.start_run("training_pipeline") if tracker else _noop_context()

    with mlflow_context:
        # =====================================================================
        # Step 1: Load Data
        # =====================================================================
        logger.info("\n📌 STEP 1: Loading Dataset")
        logger.info("─" * 40)
        df = load_dataset()
        summary = get_dataset_summary(df)

        if tracker:
            tracker.log_dataset_info(df)

        # =====================================================================
        # Step 2: EDA Visualizations
        # =====================================================================
        if not args.no_eda and not args.no_plots:
            logger.info("\n📌 STEP 2: Exploratory Data Analysis")
            logger.info("─" * 40)
            run_eda(df)
        else:
            logger.info("\n📌 STEP 2: EDA skipped")

        # =====================================================================
        # Step 3: Preprocessing
        # =====================================================================
        logger.info("\n📌 STEP 3: Preprocessing")
        logger.info("─" * 40)
        data = preprocess_pipeline(df)

        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        scaler = data["scaler"]

        # =====================================================================
        # Step 4: Class Balancing (SMOTE)
        # =====================================================================
        if not args.no_smote:
            logger.info("\n📌 STEP 4: Class Balancing")
            logger.info("─" * 40)
            X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
            if tracker:
                tracker.log_params({"smote": "enabled"})
        else:
            logger.info("\n📌 STEP 4: SMOTE skipped")
            X_train_balanced = X_train
            y_train_balanced = y_train
            if tracker:
                tracker.log_params({"smote": "disabled"})

        # =====================================================================
        # Step 5: Hyperparameter Optimization (optional)
        # =====================================================================
        n_legit = (y_train == 0).sum()
        n_fraud = (y_train == 1).sum()
        scale_pos_weight = n_legit / n_fraud

        optimized_params = {}

        if args.optimize:
            logger.info("\n📌 STEP 5: Hyperparameter Optimization")
            logger.info("─" * 40)
            from src.optimizer import run_optimization, cross_validate_model

            for model_type in ["random_forest", "xgboost"]:
                spw = scale_pos_weight if model_type == "xgboost" else None
                result = run_optimization(
                    X_train_balanced, y_train_balanced,
                    model_type=model_type,
                    engine=args.engine,
                    n_trials=args.n_trials,
                    cv_folds=args.cv_folds,
                    scale_pos_weight=spw,
                )
                optimized_params[model_type] = result["best_params"]

                if tracker:
                    tracker.log_params({
                        f"optim_{model_type}_best_score": f"{result['best_score']:.4f}",
                    })
        else:
            logger.info("\n📌 STEP 5: Optimization skipped (use --optimize)")

        # =====================================================================
        # Step 6: Model Training
        # =====================================================================
        logger.info("\n📌 STEP 6: Model Training")
        logger.info("─" * 40)

        if optimized_params:
            models = {}
            for model_type, params in optimized_params.items():
                display_name = "Random Forest" if model_type == "random_forest" else "XGBoost"
                spw = scale_pos_weight if model_type == "xgboost" else None
                models[display_name] = create_model_from_params(
                    model_type, params, scale_pos_weight=spw
                )
        else:
            models = get_all_models(scale_pos_weight=scale_pos_weight)

        trained_models = {}
        training_times = {}

        for name, model in models.items():
            trained_model, train_time = train_model(
                model, X_train_balanced, y_train_balanced, model_name=name
            )
            trained_models[name] = trained_model
            training_times[name] = train_time

            if tracker:
                tracker.log_metrics({
                    f"{name.lower().replace(' ', '_')}_train_time": train_time,
                })
                tracker.log_params({
                    f"{name.lower().replace(' ', '_')}_params": str(model.get_params()),
                })

        # =====================================================================
        # Step 7: Cross-Validation (if optimizer was used)
        # =====================================================================
        if args.optimize:
            logger.info("\n📌 STEP 7: Cross-Validation")
            logger.info("─" * 40)
            from src.optimizer import cross_validate_model

            for name, model_obj in trained_models.items():
                cv_result = cross_validate_model(
                    model_obj, X_train_balanced, y_train_balanced,
                    cv_folds=args.cv_folds,
                )
                if tracker:
                    tracker.log_cv_results(cv_result)
        else:
            logger.info("\n📌 STEP 7: CV skipped (runs with --optimize)")

        # =====================================================================
        # Step 8: Model Evaluation
        # =====================================================================
        logger.info("\n📌 STEP 8: Model Evaluation")
        logger.info("─" * 40)

        all_metrics = []
        for name, model in trained_models.items():
            metrics = evaluate_model(model, X_test, y_test, model_name=name)
            metrics["training_time"] = training_times[name]
            print_metrics(metrics)
            all_metrics.append(metrics)

            if tracker:
                prefix = name.lower().replace(" ", "_")
                tracker.log_metrics({
                    f"{prefix}_accuracy": metrics["accuracy"],
                    f"{prefix}_precision": metrics["precision"],
                    f"{prefix}_recall": metrics["recall"],
                    f"{prefix}_f1": metrics["f1_score"],
                    f"{prefix}_auc_roc": metrics["auc_roc"],
                    f"{prefix}_auc_pr": metrics["auc_pr"],
                })

        comparison_df = compare_models(all_metrics)
        best = get_best_model(all_metrics, metric="f1_score")

        # =====================================================================
        # Step 9: Threshold Optimization
        # =====================================================================
        if not args.no_threshold:
            logger.info("\n📌 STEP 9: Threshold Optimization")
            logger.info("─" * 40)

            for name, model in trained_models.items():
                y_proba = model.predict_proba(X_test)[:, 1]
                optimizer = ThresholdOptimizer()
                thresh_result = optimizer.find_optimal(y_test, y_proba)

                safe_name = name.lower().replace(" ", "_")
                optimizer.save(safe_name)

                if not args.no_plots:
                    optimizer.plot_threshold_curve()

                # Re-evaluate with optimal threshold
                opt_metrics = evaluate_model(
                    model, X_test, y_test, model_name=f"{name} (optimized)",
                    threshold=thresh_result["optimal_threshold"]
                )
                print_metrics(opt_metrics)

                if tracker:
                    tracker.log_threshold(thresh_result)
                    tracker.log_metrics({
                        f"{safe_name}_opt_f1": opt_metrics["f1_score"],
                        f"{safe_name}_opt_precision": opt_metrics["precision"],
                        f"{safe_name}_opt_recall": opt_metrics["recall"],
                    })
        else:
            logger.info("\n📌 STEP 9: Threshold optimization skipped")

        # =====================================================================
        # Step 10: Save Models & Artifacts
        # =====================================================================
        logger.info("\n📌 STEP 10: Saving Models")
        logger.info("─" * 40)

        for name, model in trained_models.items():
            idx = list(trained_models.keys()).index(name)
            save_metrics = {
                "accuracy": all_metrics[idx]["accuracy"],
                "precision": all_metrics[idx]["precision"],
                "recall": all_metrics[idx]["recall"],
                "f1_score": all_metrics[idx]["f1_score"],
                "auc_roc": all_metrics[idx]["auc_roc"],
                "auc_pr": all_metrics[idx]["auc_pr"],
            }
            save_model(model, name, scaler=scaler, metrics=save_metrics)

            if tracker:
                tracker.log_model(model, name.lower().replace(" ", "_"))

        # =====================================================================
        # Step 11: SHAP Explainability
        # =====================================================================
        if SHAP_ENABLED and not args.no_shap:
            logger.info("\n📌 STEP 11: SHAP Explainability")
            logger.info("─" * 40)
            try:
                from src.explainability import FraudExplainer

                feature_names = FEATURE_COLS
                if isinstance(X_test, np.ndarray):
                    feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]

                for name, model in trained_models.items():
                    logger.info("Computing SHAP for %s...", name)
                    explainer = FraudExplainer(model, model_name=name)

                    if not args.no_plots:
                        # Use a sample of test data for SHAP plots
                        X_sample = X_test[:500] if len(X_test) > 500 else X_test
                        explainer.plot_summary(X_sample, feature_names=feature_names)
                        explainer.plot_bar_importance(X_sample, feature_names=feature_names)

                    explainer.save_explainer()

                    if tracker:
                        for fig_file in FIGURES_DIR.glob(f"shap_*{name.lower().replace(' ', '_')}*"):
                            tracker.log_artifact(fig_file)

            except ImportError:
                logger.warning("SHAP not installed. Skipping. Install with: pip install shap")
            except Exception as e:
                logger.warning("SHAP analysis failed: %s", e)
        else:
            logger.info("\n📌 STEP 11: SHAP skipped")

        # =====================================================================
        # Step 12: Evaluation Visualizations
        # =====================================================================
        if not args.no_plots:
            logger.info("\n📌 STEP 12: Generating Evaluation Plots")
            logger.info("─" * 40)

            feature_names = FEATURE_COLS
            if isinstance(X_train, np.ndarray):
                feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

            plot_all_results(all_metrics, trained_models, feature_names)

            if tracker:
                tracker.log_artifacts_dir(str(FIGURES_DIR))
        else:
            logger.info("\n📌 STEP 12: Plots skipped")

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - start_time

    logger.info("═" * 60)
    logger.info("  ✅ TRAINING PIPELINE COMPLETE")
    logger.info("═" * 60)
    logger.info("  Total time:        %.2fs", total_time)
    logger.info("  Best model:        %s", best["model_name"])
    logger.info("  Best F1-Score:     %.4f", best["f1_score"])
    logger.info("  Best AUC-ROC:      %.4f", best["auc_roc"])
    logger.info("  Models saved to:   %s", MODELS_DIR)
    logger.info("  Figures saved to:  %s", FIGURES_DIR)
    logger.info("")
    logger.info("  Next steps:")
    logger.info("    • Run predictions:  python predict.py --sample random")
    logger.info("    • Launch API:       uvicorn api.main:app --reload")
    logger.info("    • Launch web app:   streamlit run app.py")
    if tracker:
        logger.info("    • View MLflow:      mlflow ui")
    logger.info("═" * 60)


class _noop_context:
    """No-op context manager."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


if __name__ == "__main__":
    main()
