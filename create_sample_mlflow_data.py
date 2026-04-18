#!/usr/bin/env python3
"""
Script to create sample MLflow experiments for testing.
"""

import mlflow
import numpy as np
from datetime import datetime

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5001")

# Create sample experiment
experiment_name = "skin-cancer-classification"
mlflow.set_experiment(experiment_name)

# Create sample runs
models = ["efficientnet", "resnet50", "vit"]
for model_name in models:
    with mlflow.start_run(run_name=f"{model_name}_training"):

        # Log parameters
        mlflow.log_params({
            "model": model_name,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "optimizer": "Adam"
        })

        # Log sample metrics
        for epoch in range(10):
            train_loss = np.random.uniform(0.1, 0.5)
            val_accuracy = np.random.uniform(0.7, 0.95)
            val_f1 = np.random.uniform(0.65, 0.9)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_accuracy": val_accuracy,
                "val_f1_score": val_f1
            }, step=epoch)

        # Log final metrics
        mlflow.log_metrics({
            "best_val_accuracy": 0.92,
            "best_val_f1_score": 0.88,
            "final_train_loss": 0.15
        })

        # Log sample artifact (create a dummy file)
        with open(f"sample_{model_name}_metrics.txt", "w") as f:
            f.write(f"Model: {model_name}\nBest accuracy: 0.92\nBest F1: 0.88\n")

        mlflow.log_artifact(f"sample_{model_name}_metrics.txt", "metrics")

        print(f"Created run for {model_name}")

print("Sample MLflow experiments created successfully!")
print("Check http://localhost:5001 to view experiments.")