"""
FastAPI Application
===================
REST API for credit card fraud detection with prediction, batch processing,
SHAP explanations, and model management endpoints.

Run: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import os
import io
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from api.schemas import (
    TransactionRequest, PredictionResponse,
    BatchPredictionResponse, BatchPredictionResult,
    ExplanationResponse, FeatureContribution,
    HealthResponse, ModelInfoResponse, ThresholdResponse,
)
from api.dependencies import get_predictor, reload_predictor, get_explainer
from src.config import API_CORS_ORIGINS, FEATURE_COLS
from src.logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# App Setup
# =============================================================================

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Production-grade REST API for real-time fraud detection using "
        "XGBoost and Random Forest models with SHAP explainability."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Startup Event
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Preload model on startup."""
    try:
        get_predictor()
        logger.info("🚀 API started — model preloaded")
    except Exception as e:
        logger.error("Failed to preload model: %s", e)


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model status."""
    try:
        predictor = get_predictor()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=predictor.model_name,
            version="2.0.0",
        )
    except Exception:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_name="none",
            version="2.0.0",
        )


# =============================================================================
# Prediction Endpoints
# =============================================================================

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(
    transaction: TransactionRequest,
    model: str = Query(default=None, description="Model name (xgboost or random_forest)"),
    threshold: float = Query(default=None, ge=0.0, le=1.0, description="Custom threshold"),
):
    """
    Predict whether a single transaction is fraudulent.

    Uses the optimal threshold by default. Override with the `threshold` parameter.
    """
    try:
        predictor = get_predictor(model)
        transaction_dict = transaction.model_dump()
        result = predictor.predict_single(transaction_dict, threshold=threshold)
        return PredictionResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    file: UploadFile = File(..., description="CSV file with transaction data"),
    model: str = Query(default=None, description="Model name"),
    threshold: float = Query(default=None, ge=0.0, le=1.0, description="Custom threshold"),
):
    """
    Predict fraud for a batch of transactions uploaded as CSV.

    The CSV must contain all required feature columns (Time, V1-V28, Amount).
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        missing = set(FEATURE_COLS) - set(df.columns)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {sorted(missing)}"
            )

        predictor = get_predictor(model)
        results_df = predictor.predict_batch(df, threshold=threshold)

        n_fraud = int(results_df["Is_Fraud"].sum())
        n_legit = len(results_df) - n_fraud

        results = []
        for idx, row in results_df.iterrows():
            results.append(BatchPredictionResult(
                index=int(idx),
                prediction=row["Prediction"],
                is_fraud=bool(row["Is_Fraud"]),
                fraud_probability=float(row["Fraud_Probability"]),
                confidence=float(row["Confidence"]),
                amount=float(row["Amount"]) if "Amount" in row else None,
            ))

        return BatchPredictionResponse(
            total_transactions=len(results_df),
            fraudulent=n_fraud,
            legitimate=n_legit,
            fraud_rate=n_fraud / len(results_df) if len(results_df) > 0 else 0.0,
            results=results,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# =============================================================================
# Explainability Endpoints
# =============================================================================

@app.post("/predict/explain", response_model=ExplanationResponse, tags=["Explainability"])
async def predict_with_explanation(
    transaction: TransactionRequest,
    model: str = Query(default=None, description="Model name"),
    threshold: float = Query(default=None, ge=0.0, le=1.0, description="Custom threshold"),
):
    """
    Predict with SHAP explanation showing top contributing features.
    """
    try:
        predictor = get_predictor(model)
        transaction_dict = transaction.model_dump()

        # Get prediction
        pred_result = predictor.predict_single(transaction_dict, threshold=threshold)

        # Get SHAP explanation
        explainer = get_explainer(model)
        if explainer is None:
            raise HTTPException(
                status_code=503,
                detail="SHAP explainer not available. Ensure SHAP is installed and model is trained."
            )

        explanation = explainer.explain_single(transaction_dict)

        top_features = [
            FeatureContribution(**f) for f in explanation["top_features"]
        ]

        return ExplanationResponse(
            prediction=PredictionResponse(**pred_result),
            base_value=explanation["base_value"],
            top_features=top_features,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Explanation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


# =============================================================================
# Model Management Endpoints
# =============================================================================

@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info(
    model: str = Query(default=None, description="Model name"),
):
    """Get information about the currently loaded model."""
    try:
        predictor = get_predictor(model)
        info = predictor.get_model_info()
        return ModelInfoResponse(
            model_name=info["model_name"],
            model_type=info["model_type"],
            has_scaler=info["has_scaler"],
            n_features=info["n_features"],
            optimal_threshold=info["optimal_threshold"],
            training_metrics=info.get("training_metrics"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/threshold", response_model=ThresholdResponse, tags=["Model"])
async def get_threshold_info(
    model: str = Query(default=None, description="Model name"),
):
    """Get the current optimal threshold and all strategy results."""
    try:
        predictor = get_predictor(model)
        from src.threshold import ThresholdOptimizer
        thresh_data = ThresholdOptimizer.load(predictor.model_name)

        return ThresholdResponse(
            model_name=predictor.model_name,
            optimal_threshold=predictor.optimal_threshold,
            strategy=thresh_data.get("strategy") if thresh_data else None,
            all_strategies=thresh_data.get("all_results") if thresh_data else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/model/reload", tags=["Model"])
async def reload_model():
    """Hot-reload model artifacts without restarting the server."""
    try:
        reload_predictor()
        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


# =============================================================================
# Root
# =============================================================================

@app.get("/", tags=["System"])
async def root():
    """API root — redirects to docs."""
    return {
        "message": "Credit Card Fraud Detection API",
        "docs": "/docs",
        "health": "/health",
    }
