# 🏥 Medical Image Classification — Skin Cancer Detection

![CI/CD](https://github.com/YOUR_ORG/medical-image-classification/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end ML system for classifying dermoscopic images of skin lesions using the **ISIC Skin Cancer dataset**. The system assists dermatologists in detecting potential malignancies by comparing multiple deep learning architectures and deploying the best model as a production-ready API.

> **⚠️ Disclaimer**: This is a decision-support tool for educational purposes. It is NOT a medical device and should not be used for clinical diagnosis without professional medical review.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Models](#models)
- [Quick Start](#quick-start)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [API Usage](#api-usage)
- [Docker Deployment](#docker-deployment)
- [Monitoring](#monitoring)
- [Testing](#testing)
- [Responsible AI](#responsible-ai)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Problem Statement
Skin cancer is one of the most common cancers worldwide. Early detection significantly improves patient outcomes. This system uses deep learning to classify dermoscopic images into 9 categories of skin lesions, prioritizing **recall** (sensitivity) to minimize missed diagnoses.

### Success Metrics
| Level | Metric | Target |
|-------|--------|--------|
| **Business** | Missed malignancy rate | < 5% |
| **System** | API latency (p95) | < 2 seconds |
| **Model** | Recall (macro) | > 0.75 |
| **Model** | ROC-AUC (macro) | > 0.85 |

### Supported Classes
| Class | Diagnosis | Type |
|-------|-----------|------|
| actinic keratosis | Actinic Keratosis | Pre-malignant |
| basal cell carcinoma | Basal Cell Carcinoma | **Malignant** |
| dermatofibroma | Dermatofibroma | Benign |
| melanoma | Melanoma | **Malignant** |
| nevus | Melanocytic Nevus | Benign |
| pigmented benign keratosis | Pigmented Benign Keratosis | Benign |
| seborrheic keratosis | Seborrheic Keratosis | Benign |
| squamous cell carcinoma | Squamous Cell Carcinoma | **Malignant** |
| vascular lesion | Vascular Lesion | Benign |

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design, data flow diagrams, and technology decisions.

**High-Level Overview:**
```
User → FastAPI → Model Inference → Prediction + Confidence
                    ↑
        Trained Model (.pth)
                    ↑
  Data Pipeline → Training (MLflow) → Evaluation → Best Model
                    ↑
          ISIC Dataset (images + labels)
```

**Monitoring Stack:**
```
API (/metrics) → Prometheus → Grafana Dashboards
                     ↓
              Alert Rules → Notifications
```

---

## Models

We implement and compare **4 model architectures**:

| Model | Architecture | Params | Pre-trained |
|-------|-------------|--------|-------------|
| **Custom CNN** | 4-block CNN from scratch | ~2M | No |
| **ResNet50** | Transfer learning, fine-tune layer4 | ~25M | ImageNet |
| **EfficientNet-B0** | Transfer learning, fine-tune classifier | ~5M | ImageNet |
| **ViT-B/16** | Vision Transformer, fine-tune last 2 blocks | ~86M | ImageNet |

All models use:
- **Loss**: Weighted CrossEntropyLoss (class imbalance handling)
- **Optimizer**: Adam with weight decay
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: patience=7

---

## Quick Start

### Prerequisites
- Python 3.12
- Docker & Docker Compose (for deployment)
- GPU recommended (NVIDIA CUDA) but CPU works

### Installation

```bash
# Clone the repository
git clone https://github.com/tmt2504/medical-image-classification.git
cd medical-image-classification

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Windows Command Prompt (CMD):
# venv\Scripts\activate.bat

# Install dependencies
pip3 install -r requirements.txt
```

---

## Dataset Setup

### Option 1: Auto-download via kagglehub
```bash
# On Linux/macOS:
python3 -c "from src.data_pipeline import download_dataset; download_dataset()"
# On Windows:
python -c "from src.data_pipeline import download_dataset; download_dataset()"
```
This will automatically download and set up the ISIC dataset in the `data/` directory.

### Option 2: Manual download
1. Install kagglehub: `pip install kagglehub`
2. Download the dataset:
```python
import kagglehub
path = kagglehub.dataset_download("nodoubttome/skin-cancer9-classesisic")
print("Path to dataset files:", path)
```
3. Copy or symlink the dataset into the `data/` directory

### Expected structure
```
data/
└── Skin cancer ISIC The International Skin Imaging Collaboration/
    ├── Train/
    │   ├── actinic keratosis/
    │   ├── basal cell carcinoma/
    │   ├── dermatofibroma/
    │   ├── melanoma/
    │   ├── nevus/
    │   ├── pigmented benign keratosis/
    │   ├── seborrheic keratosis/
    │   ├── squamous cell carcinoma/
    │   └── vascular lesion/
    └── Test/
        └── (same 9 class folders)
```

4. Verify data quality:
```bash
python3 -c "from src.data_pipeline import validate_data_quality; print(validate_data_quality('data'))"
```

---

## Training

### Train a single model
```bash
python3 -m src.train --model resnet50 --data-dir data --epochs 30 --batch-size 32 --lr 1e-4
```

### Train all models
```bash
python3 -m src.train --model all --data-dir data --epochs 30
```

### Available models
- `custom_cnn` — Custom CNN from scratch
- `resnet50` — ResNet50 transfer learning
- `efficientnet` — EfficientNet-B0 transfer learning
- `vit` — Vision Transformer (ViT-B/16)

### MLflow Tracking
Training metrics are automatically logged to MLflow:
```bash
mlflow ui --port 5000
```
Open http://localhost:5000 to view experiments.

---

## Evaluation

### Evaluate all trained models
```bash
python3 -m src.evaluate --data-dir data --models-dir models --output-dir results
```

### Generated outputs (in `results/`):
- `model_comparison.csv` — Metrics comparison table
- `model_comparison.txt` — Detailed analysis with recommendation
- `model_comparison_chart.png` — Bar chart comparison
- `confusion_matrix_*.png` — Per-model confusion matrices
- `roc_curves_*.png` — Per-model ROC curves
- `classification_report_*.txt` — Per-model classification reports

---

## API Usage

### Start the API server
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Swagger Documentation
Open http://localhost:8000/docs for interactive API docs.

### Predict endpoint
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/skin_image.jpg"
```

### Response format
```json
{
  "predicted_class": "nevus",
  "confidence": 0.8523,
  "class_probabilities": {
    "actinic keratosis": 0.0121,
    "basal cell carcinoma": 0.0234,
    "dermatofibroma": 0.0098,
    "melanoma": 0.0521,
    "nevus": 0.8523,
    "pigmented benign keratosis": 0.0112,
    "seborrheic keratosis": 0.0089,
    "squamous cell carcinoma": 0.0179,
    "vascular lesion": 0.0123
  },
  "model_name": "efficientnet",
  "is_malignant": false,
  "risk_level": "LOW"
}
```

### Other endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/classes` | Supported classes |
| GET | `/docs` | Swagger UI |
| GET | `/metrics` | Prometheus metrics |
| POST | `/predict` | Classify image |

---

## Docker Deployment

### Build and run with Docker Compose
```bash
# Build and start all services
docker-compose up -d --build

# Check status
docker-compose ps

# View API logs
docker-compose logs -f api
```

### Services
| Service | Port | Description |
|---------|------|-------------|
| api | 8000 | FastAPI prediction server |
| prometheus | 9090 | Metrics collection |
| grafana | 3000 | Monitoring dashboards |
| mlflow | 5000 | Experiment tracking |

### Run API only
```bash
docker build -t skin-cancer-api .
docker run -p 8000:8000 -v ./models:/app/models:ro skin-cancer-api
```

---

## Monitoring

### Prometheus Metrics
Available at http://localhost:9090 (when running via Docker Compose).

Key metrics:
- `skin_cancer_predictions_total` — Total predictions by class
- `skin_cancer_prediction_latency_seconds` — Prediction latency
- `skin_cancer_prediction_confidence` — Confidence distribution
- `skin_cancer_malignant_predictions_total` — Malignant predictions count
- `skin_cancer_low_confidence_predictions_total` — Low-confidence predictions

### Grafana Dashboards
Available at http://localhost:3000 (admin/admin).

Dashboard includes:
- Prediction rate and total count
- Latency percentiles (p50, p95)
- Class distribution pie chart
- Confidence histogram
- Error rate gauge
- Malignant prediction tracking

### Alerting Rules
- **HighErrorRate**: API error rate > 5% for 2 minutes
- **HighLatency**: p95 latency > 5 seconds for 3 minutes
- **LowConfidencePredictions**: High rate of uncertain predictions (drift indicator)
- **APIDown**: API unreachable for 1 minute
- **HighMalignantRate**: Unusual malignant prediction pattern

---

## Testing

### Run all tests
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Run specific test suites
```bash
# Unit tests (models)
pytest tests/test_model.py -v

# API tests
pytest tests/test_api.py -v

# Data quality tests
pytest tests/test_data_quality.py -v

# Integration tests
pytest tests/test_integration.py -v
```

### Test categories
| Test File | Type | What it tests |
|-----------|------|---------------|
| `test_model.py` | Unit | Model architectures, output shapes, gradients |
| `test_api.py` | Unit/Integration | API endpoints, error handling, OpenAPI spec |
| `test_data_quality.py` | Data Quality | Transforms, dataset integrity, class weights |
| `test_integration.py` | Integration | End-to-end inference, model validation |

---

## Responsible AI

### Explainability
- **Grad-CAM** visualizations for CNN, ResNet50, and EfficientNet
- **Attention maps** for Vision Transformer
- See `src/explainability.py` for implementation

### Fairness
- Per-class performance analysis to detect disparities
- Recall gap monitoring across disease types
- Equalized odds evaluation

### Data Privacy (HIPAA)
- De-identified images only (ISIC dataset)
- No patient PII processed or stored
- Local inference; no data leaves the system

### Ethics
- Designed as decision _support_, not replacement for dermatologists
- Optimized for recall to minimize missed diagnoses
- Confidence thresholds flag uncertain predictions for human review
- Full report: `results/responsible_ai_report.txt`

---

## Project Structure

```
medical-image-classification/
├── .github/workflows/
│   └── ci.yml                    # CI/CD pipeline
├── dashboards/
│   ├── datasources.yml           # Grafana datasource config
│   ├── dashboards.yml            # Grafana dashboard provisioning
│   └── skin_cancer_dashboard.json # Grafana dashboard
├── data/                         # Dataset (not in git)
├── models/                       # Saved model checkpoints
├── prometheus/
│   ├── prometheus.yml            # Prometheus config
│   └── alert_rules.yml           # Alert rules
├── src/
│   ├── models/
│   │   ├── custom_cnn.py         # Custom CNN architecture
│   │   ├── transfer_learning.py  # ResNet50 + EfficientNet
│   │   └── vit_model.py          # Vision Transformer
│   ├── api.py                    # FastAPI REST API
│   ├── data_pipeline.py          # Data loading & augmentation
│   ├── evaluate.py               # Model evaluation & comparison
│   ├── explainability.py         # Grad-CAM & fairness analysis
│   ├── inference.py              # Single-image prediction
│   ├── monitoring.py             # Prometheus metrics
│   └── train.py                  # Training with MLflow
├── tests/
│   ├── test_api.py               # API tests
│   ├── test_data_quality.py      # Data quality tests
│   ├── test_integration.py       # Integration tests
│   └── test_model.py             # Model unit tests
├── ARCHITECTURE.md               # System design documentation
├── CONTRIBUTING.md               # Team roles & contribution guide
├── Dockerfile                    # Multi-stage Docker build
├── docker-compose.yml            # Multi-service orchestration
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `--batch-size` (try 16 or 8) |
| Model checkpoint not found | Run training first: `python -m src.train` |
| Slow training on CPU | Use GPU or reduce epochs; ViT is especially slow on CPU |
| Docker build fails | Ensure Docker is running and you have enough disk space |
| Import errors | Activate venv and verify `pip install -r requirements.txt` |
| ISIC data not found | Check data directory structure matches expected format |

---

## License

This project is for educational purposes as part of DDM501 - AI in Production.
