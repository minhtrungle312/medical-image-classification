"""
API endpoint tests.

Tests the FastAPI prediction endpoint, health check,
error handling, and response format.
"""

import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image

from src.api import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create a sample image as bytes for upload."""
    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def mock_predictor():
    """Mock the predictor to avoid loading real model weights."""
    mock = MagicMock()
    mock.model_name = "efficientnet"
    mock.predict.return_value = {
        "predicted_class": "nevus",
        "confidence": 0.85,
        "class_probabilities": {
            "actinic keratosis": 0.01,
            "basal cell carcinoma": 0.02,
            "dermatofibroma": 0.01,
            "melanoma": 0.05,
            "nevus": 0.85,
            "pigmented benign keratosis": 0.02,
            "seborrheic keratosis": 0.01,
            "squamous cell carcinoma": 0.02,
            "vascular lesion": 0.01,
        },
        "model_name": "efficientnet",
        "is_malignant": False,
        "risk_level": "LOW",
    }
    return mock


class TestHealthEndpoint:
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data

    def test_health_no_model(self, client):
        """Health check when model is not loaded."""
        import src.api

        original = src.api.predictor
        src.api.predictor = None
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        src.api.predictor = original

    def test_health_with_model(self, client, mock_predictor):
        import src.api

        original = src.api.predictor
        src.api.predictor = mock_predictor
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        src.api.predictor = original

    def test_classes_endpoint(self, client):
        response = client.get("/classes")
        assert response.status_code == 200
        data = response.json()
        assert "classes" in data
        assert "melanoma" in data["classes"]


class TestPredictEndpoint:
    def test_predict_success(self, client, sample_image_bytes, mock_predictor):
        import src.api

        original = src.api.predictor
        src.api.predictor = mock_predictor

        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )

        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "class_probabilities" in data
        assert "risk_level" in data
        src.api.predictor = original

    def test_predict_invalid_file_type(self, client, mock_predictor):
        import src.api

        original = src.api.predictor
        src.api.predictor = mock_predictor

        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400
        src.api.predictor = original

    def test_predict_no_model(self, client, sample_image_bytes):
        import src.api

        original = src.api.predictor
        src.api.predictor = None

        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 503
        src.api.predictor = original

    def test_predict_no_file(self, client, mock_predictor):
        import src.api

        original = src.api.predictor
        src.api.predictor = mock_predictor

        response = client.post("/predict")
        assert response.status_code == 422  # Validation error
        src.api.predictor = original

    def test_predict_corrupt_image(self, client, mock_predictor):
        import src.api

        original = src.api.predictor
        src.api.predictor = mock_predictor

        response = client.post(
            "/predict",
            files={"file": ("test.jpg", b"corrupt data", "image/jpeg")},
        )
        assert response.status_code == 400
        src.api.predictor = original


class TestOpenAPIDoc:
    def test_openapi_available(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/predict" in schema["paths"]

    def test_docs_available(self, client):
        response = client.get("/docs")
        assert response.status_code == 200
