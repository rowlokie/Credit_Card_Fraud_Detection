"""
Pydantic Schemas
================
Request/response models for the FastAPI endpoints with validation.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Single transaction input for prediction."""
    Time: float = Field(..., ge=0, description="Seconds since first transaction")
    V1: float = Field(default=0.0)
    V2: float = Field(default=0.0)
    V3: float = Field(default=0.0)
    V4: float = Field(default=0.0)
    V5: float = Field(default=0.0)
    V6: float = Field(default=0.0)
    V7: float = Field(default=0.0)
    V8: float = Field(default=0.0)
    V9: float = Field(default=0.0)
    V10: float = Field(default=0.0)
    V11: float = Field(default=0.0)
    V12: float = Field(default=0.0)
    V13: float = Field(default=0.0)
    V14: float = Field(default=0.0)
    V15: float = Field(default=0.0)
    V16: float = Field(default=0.0)
    V17: float = Field(default=0.0)
    V18: float = Field(default=0.0)
    V19: float = Field(default=0.0)
    V20: float = Field(default=0.0)
    V21: float = Field(default=0.0)
    V22: float = Field(default=0.0)
    V23: float = Field(default=0.0)
    V24: float = Field(default=0.0)
    V25: float = Field(default=0.0)
    V26: float = Field(default=0.0)
    V27: float = Field(default=0.0)
    V28: float = Field(default=0.0)
    Amount: float = Field(..., ge=0, description="Transaction amount in dollars")

    model_config = {"json_schema_extra": {
        "examples": [{
            "Time": 406.0, "V1": -0.24, "V2": 0.70, "V3": 0.53,
            "V4": 0.24, "V5": -0.28, "V6": -0.17, "V7": -0.05,
            "V8": 0.09, "V9": 0.23, "V10": -0.11, "V11": -0.63,
            "V12": -0.08, "V13": -0.31, "V14": 0.39, "V15": -0.45,
            "V16": 0.09, "V17": 0.27, "V18": -0.08, "V19": -0.09,
            "V20": -0.07, "V21": 0.10, "V22": 0.30, "V23": -0.01,
            "V24": 0.02, "V25": 0.31, "V26": 0.04, "V27": 0.04,
            "V28": 0.01, "Amount": 44.50,
        }]
    }}


class PredictionResponse(BaseModel):
    """Single prediction result."""
    prediction: str
    is_fraud: bool
    confidence: float
    fraud_probability: float
    legitimate_probability: float
    threshold_used: float


class BatchPredictionResult(BaseModel):
    """Single row in batch prediction."""
    index: int
    prediction: str
    is_fraud: bool
    fraud_probability: float
    confidence: float
    amount: Optional[float] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    total_transactions: int
    fraudulent: int
    legitimate: int
    fraud_rate: float
    results: List[BatchPredictionResult]


class FeatureContribution(BaseModel):
    """Single feature SHAP contribution."""
    feature: str
    shap_value: float
    abs_shap_value: float
    feature_value: float


class ExplanationResponse(BaseModel):
    """Prediction with SHAP explanation."""
    prediction: PredictionResponse
    base_value: float
    top_features: List[FeatureContribution]


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    model_loaded: bool
    model_name: str
    version: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    model_type: str
    has_scaler: bool
    n_features: int
    optimal_threshold: float
    training_metrics: Optional[Dict] = None


class ThresholdResponse(BaseModel):
    """Threshold information response."""
    model_name: str
    optimal_threshold: float
    strategy: Optional[str] = None
    all_strategies: Optional[Dict[str, float]] = None
