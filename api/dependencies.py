"""
Dependencies Module
===================
FastAPI dependency injection for model loading and singleton management.
"""

import sys
import os
from functools import lru_cache

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predictor import FraudPredictor
from src.config import API_DEFAULT_MODEL
from src.logging_config import get_logger

logger = get_logger(__name__)

# Global predictor instance
_predictor_instance = None
_current_model_name = None


def get_predictor(model_name: str = None) -> FraudPredictor:
    """
    Get or create the FraudPredictor singleton.

    Parameters
    ----------
    model_name : str, optional
        Model to load. Defaults to API_DEFAULT_MODEL.

    Returns
    -------
    FraudPredictor
        Loaded predictor instance.
    """
    global _predictor_instance, _current_model_name

    model_name = model_name or API_DEFAULT_MODEL

    if _predictor_instance is None or _current_model_name != model_name:
        logger.info("Loading model: %s", model_name)
        _predictor_instance = FraudPredictor(model_name=model_name)
        _current_model_name = model_name
        logger.info("Model loaded successfully: %s", model_name)

    return _predictor_instance


def reload_predictor():
    """Force reload the current predictor."""
    global _predictor_instance
    if _predictor_instance:
        _predictor_instance.reload()
        logger.info("Model reloaded successfully")
    return _predictor_instance


def get_explainer(model_name: str = None):
    """
    Get a SHAP explainer for the current model.

    Returns
    -------
    FraudExplainer or None
    """
    try:
        from src.explainability import FraudExplainer
        predictor = get_predictor(model_name)
        return FraudExplainer(predictor.model, model_name=predictor.model_name)
    except Exception as e:
        logger.warning("Failed to create SHAP explainer: %s", e)
        return None
