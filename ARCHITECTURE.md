# System Architecture — Skin Cancer Classification

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER / CLIENT                                │
│                    (Browser / curl / mobile)                        │
└─────────────────────┬───────────────────────────────────────────────┘
                      │ HTTP POST /predict (image)
                      ▼
┌────────────────────────────────────────────────────────────────────┐
│                      FASTAPI REST API                              │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │  /predict   │  │  /health     │  │  /metrics (Prometheus)    │  │
│  │  /classes   │  │  /docs       │  │                           │  │
│  └──────┬──────┘  └──────────────┘  └───────────────────────────┘  │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────────────────┐                                      │
│  │  INFERENCE ENGINE        │                                      │
│  │  - Image preprocessing   │                                      │
│  │  - Model forward pass    │                                      │
│  │  - Post-processing       │                                      │
│  │  - Risk assessment       │                                      │
│  └──────────┬───────────────┘                                      │
│             │                                                      │
│             ▼                                                      │
│  ┌──────────────────────────┐                                      │
│  │  TRAINED MODEL (.pth)    │                                      │
│  │  (EfficientNet-B0)       │                                      │
│  └──────────────────────────┘                                      │
└────────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MONITORING STACK                                │
│  ┌────────────┐    ┌────────────┐    ┌────────────────────────┐     │
│  │ Prometheus │───▶│  Grafana   │    │  Alerting Rules        │     │
│  │ (metrics)  │    │ (dashboard)│    │  (error/latency/drift) │     │
│  └────────────┘    └────────────┘    └────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. ML Pipeline Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐
│ ISIC Dataset │────▶│ Data Pipeline│────▶│ Preprocessed Data    │
│ (raw images) │     │  - Scan dirs │     │ - train/val/test     │
│              │     │  - Split     │     │ - augmented          │
│              │     │  - Augment   │     │ - weighted sampling  │
└──────────────┘     └──────────────┘     └──────────┬───────────┘
                                                     │
                        ┌────────────────────────────┘
                        ▼
              ┌──────────────────┐
              │  MODEL TRAINING  │
              │  ┌──────────────┐│
              │  │ ResNet50     ││  ──▶  MLflow Tracking
              │  │ EfficientNet ││       (params, metrics,
              │  │ ViT-B/16     ││        artifacts)
              │  └──────────────┘│
              │  - Weighted loss │
              │  - Adam optimizer│
              │  - Early stopping│
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  EVALUATION      │
              │  - Accuracy      │
              │  - Precision     │
              │  - Recall ★      │
              │  - F1-Score      │
              │  - ROC-AUC       │
              │  - Confusion Mat.│
              │  - ROC Curves    │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  MODEL COMPARISON│
              │  & SELECTION     │──▶ Best model deployed
              │  (recall-focused)│
              └──────────────────┘
```

## 3. Data Flow

### 3.1 Training Data Flow
```
Raw ISIC Images (JPEG) ──────────────────────────────────────────────▶
    │
    ├─▶ Load image paths + CSV labels
    ├─▶ Stratified split (85% train / 15% val)
    ├─▶ Training augmentation:
    │     - Random crop 
    │     - Horizontal/vertical flip
    │     - Random rotation
    │     - Random affine translation
    ├─▶ ImageNet normalization 
    ├─▶ Weighted random sampling (class imbalance)
    └─▶ DataLoader (batch_size=32, pin_memory=True)
```

### 3.2 Inference Data Flow
```
Input Image (JPEG/PNG) ─▶ Resize(224) ─▶ Normalize ─▶ Model ─▶ Softmax
                                                                   │
    ┌──────────────────────────────────────────────────────────────┘
    ▼
 { class: "NV", confidence: 0.85, risk: "LOW", probabilities: {...} }
```

### 3.3 Monitoring Data Flow
```
API Request ─▶ Prediction ─▶ Prometheus Metrics
                                    │
              ┌─────────────────────┘
              ├─▶ prediction_count (by class, risk)
              ├─▶ prediction_latency (histogram)
              ├─▶ confidence_distribution (histogram)
              ├─▶ malignant_count
              └─▶ low_confidence_count
                        │
                        ▼
                   Grafana ─▶ Dashboards + Alerts
```

## 4. Component Design

| Component | Responsibility | Key Interfaces |
|-----------|---------------|----------------|
| `data_pipeline.py` | Data loading, augmentation, splitting, class weights | `create_data_loaders()`, `validate_data_quality()`, `download_dataset()` |
| `models/transfer_learning.py` | ResNet50 + EfficientNet | `forward()`, `get_target_layer()`, `unfreeze_all()` |
| `models/vit_model.py` | Vision Transformer | `forward()`, `get_attention_maps()` |
| `train.py` | Training loop with MLflow | `train_model()`, `train_all_models()` |
| `evaluate.py` | Evaluation + comparison | `evaluate_all_models()`, `create_comparison_table()` |
| `inference.py` | Single-image prediction | `SkinCancerPredictor.predict()` |
| `explainability.py` | Grad-CAM + fairness | `generate_gradcam()`, `analyze_fairness()` |
| `api.py` | REST API | `POST /predict`, `GET /health` |
| `monitoring.py` | Prometheus metrics | `track_prediction()`, `setup_prometheus()` |

## 5. Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| ML Framework | PyTorch 2.0+ | Industry standard, dynamic graphs, strong ecosystem |
| Model Hub | timm, torchvision | Pre-trained weights, wide model variety |
| API Framework | FastAPI | Async, auto-docs (OpenAPI), type validation |
| Experiment Tracking | MLflow | Open source, parameter/metric/artifact logging |
| Containerization | Docker | Reproducible environments, easy deployment |
| Orchestration | Docker Compose | Multi-service management, health checks |
| Metrics | Prometheus | Pull-based, de facto standard for monitoring |
| Dashboards | Grafana | Rich visualizations, alerting integration |
| CI/CD | GitHub Actions | Native GitHub integration, free for public repos |
| Explainability | pytorch-grad-cam | State-of-art visual explanations for CNNs/ViTs |

## 6. Trade-offs Analysis

| Decision | Alternative | Why Chosen |
|----------|-------------|------------|
| PyTorch over TensorFlow | TensorFlow/Keras | Better research ecosystem, `timm` library, dynamic computation graphs |
| FastAPI over Flask | Flask, Django | Native async, auto-generated OpenAPI docs, Pydantic validation |
| Weighted loss + sampling | Only oversampling | Dual approach handles imbalance more robustly |
| ViT included | Only CNNs | Demonstrates state-of-art architecture comparison |
| Docker Compose over K8s | Kubernetes | Appropriate complexity for project scope; K8s would be over-engineering |
| SQLite MLflow backend | PostgreSQL | Simpler setup; sufficient for project scale |
| Recall as priority metric | F1 or accuracy | Medical context: false negatives (missed cancer) are more harmful than false positives |

## 7. Deployment Architecture

```
docker-compose.yml
├── api (port 8000)
│   ├── FastAPI + Uvicorn
│   ├── Trained model (volume mount)
│   └── Health check every 30s
├── prometheus (port 9090)
│   ├── Scrapes /metrics from API
│   ├── Alert rules for anomalies
│   └── 15-day data retention
├── grafana (port 3000)
│   ├── Auto-provisioned dashboards
│   ├── Prometheus datasource
│   └── Pre-built panels
└── mlflow (port 5000)
    ├── Experiment tracking UI
    ├── SQLite backend
    └── Artifact storage
```

## 8. Security Considerations

- **Non-root Docker user**: API runs as `appuser`, not root
- **Multi-stage builds**: Build dependencies not in production image
- **Read-only model volume**: Models mounted as `:ro`
- **Input validation**: File type + size checks before processing
- **No PII**: ISIC images are de-identified
- **Dependency scanning**: CI pipeline includes `safety check`
