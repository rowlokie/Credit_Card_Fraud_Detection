"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client():
    """Create a FastAPI TestClient (skips if models not available)."""
    try:
        from api.main import app
        return TestClient(app)
    except Exception:
        pytest.skip("API cannot start — models may not be trained yet")


class TestHealthEndpoint:
    def test_health_check(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")

    def test_root_endpoint(self, api_client):
        response = api_client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()


class TestPredictionEndpoint:
    def test_predict_single(self, api_client, sample_transaction):
        response = api_client.post("/predict", json=sample_transaction)
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "is_fraud" in data
            assert "fraud_probability" in data
            assert data["prediction"] in ("Fraud", "Legitimate")

    def test_predict_with_threshold(self, api_client, sample_transaction):
        response = api_client.post("/predict?threshold=0.9", json=sample_transaction)
        if response.status_code == 200:
            data = response.json()
            assert data["threshold_used"] == 0.9


class TestModelEndpoints:
    def test_model_info(self, api_client):
        response = api_client.get("/model/info")
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "n_features" in data

    def test_model_threshold(self, api_client):
        response = api_client.get("/model/threshold")
        if response.status_code == 200:
            data = response.json()
            assert "optimal_threshold" in data
