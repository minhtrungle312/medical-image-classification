"""
Monitoring module with Prometheus metrics.

Provides system and ML-specific metrics for monitoring
model performance, latency, and prediction distribution.
"""

import time
import logging
from typing import Optional

from prometheus_client import Counter, Histogram, Gauge, Info
from prometheus_fastapi_instrumentator import Instrumentator

logger = logging.getLogger(__name__)

# --- Prometheus Metrics ---

# Prediction metrics
PREDICTION_COUNTER = Counter(
    "skin_cancer_predictions_total",
    "Total number of predictions",
    ["predicted_class", "risk_level"],
)

PREDICTION_LATENCY = Histogram(
    "skin_cancer_prediction_latency_seconds",
    "Time spent processing predictions",
    buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0],
)

MODEL_CONFIDENCE = Histogram(
    "skin_cancer_prediction_confidence",
    "Distribution of prediction confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

LOW_CONFIDENCE_COUNTER = Counter(
    "skin_cancer_low_confidence_predictions_total",
    "Predictions with confidence below threshold",
)

MALIGNANT_PREDICTIONS = Counter(
    "skin_cancer_malignant_predictions_total",
    "Number of malignant predictions (MEL, BCC)",
    ["class_name"],
)

# System metrics
MODEL_INFO = Info(
    "skin_cancer_model",
    "Information about the loaded model",
)

ACTIVE_REQUESTS = Gauge(
    "skin_cancer_active_requests",
    "Number of requests currently being processed",
)

# Drift detection (simulated)
PREDICTION_DISTRIBUTION = Counter(
    "skin_cancer_class_distribution_total",
    "Distribution of predicted classes over time",
    ["class_name"],
)


def setup_prometheus(app):
    """Setup Prometheus instrumentation for FastAPI."""
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        excluded_handlers=["/metrics", "/health"],
        env_var_name="ENABLE_METRICS",
    )
    instrumentator.instrument(app).expose(app, endpoint="/metrics")
    logger.info("Prometheus metrics enabled at /metrics")


def track_prediction(predicted_class: str, confidence: float, latency: float):
    """Track a prediction in Prometheus metrics."""
    risk_level = (
        "HIGH"
        if predicted_class
        in {"melanoma", "basal cell carcinoma", "squamous cell carcinoma"}
        else "LOW"
    )

    PREDICTION_COUNTER.labels(
        predicted_class=predicted_class,
        risk_level=risk_level,
    ).inc()

    PREDICTION_LATENCY.observe(latency)
    MODEL_CONFIDENCE.observe(confidence)
    PREDICTION_DISTRIBUTION.labels(class_name=predicted_class).inc()

    if confidence < 0.5:
        LOW_CONFIDENCE_COUNTER.inc()

    if predicted_class in {
        "melanoma",
        "basal cell carcinoma",
        "squamous cell carcinoma",
    }:
        MALIGNANT_PREDICTIONS.labels(class_name=predicted_class).inc()


def set_model_info(model_name: str, num_params: int, version: str = "1.0.0"):
    """Set model metadata in Prometheus."""
    MODEL_INFO.info(
        {
            "model_name": model_name,
            "num_parameters": str(num_params),
            "version": version,
        }
    )
