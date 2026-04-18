#!/bin/bash
# Two-step training for ResNet50
# Step 1: Train with val split to find best epoch
# Step 2: Train on 100% data with that epoch count

set -e
cd "$(dirname "$0")"

export PYTHONPATH="$PWD"
export MLFLOW_TRACKING_URI="file://$PWD/mlruns"
export PYTHONUNBUFFERED=1

echo "=== ResNet50 Training Started: $(date) ==="

# ─── Step 1: Train with val split (85/15) to find best epoch ───
echo ""
echo "=== STEP 1: Training with validation split (30 epochs) ==="
echo "Started: $(date)"

python3 -m src.train --model resnet50 --data-dir data --epochs 30 --num-workers 0 2>&1 | tee /tmp/resnet50_step1.log

# Extract best epoch from the log
BEST_EPOCH=$(grep "Saved best model" /tmp/resnet50_step1.log | tail -1 | grep -oE 'epoch.*,' | grep -oE '[0-9]+' | head -1)
echo ""
echo "=== Step 1 Complete. Best model saved at epoch ${BEST_EPOCH:-unknown} ==="
echo "Finished: $(date)"

# Use best epoch count for step 2 (default to 30 if not found)
EPOCHS=${BEST_EPOCH:-30}

# ─── Step 2: Train on 100% data with best epoch count ───
echo ""
echo "=== STEP 2: Training on FULL data (${EPOCHS} epochs, --full-train) ==="
echo "Started: $(date)"

python3 -m src.train --model resnet50 --data-dir data --epochs "$EPOCHS" --num-workers 0 --full-train 2>&1 | tee /tmp/resnet50_step2.log

echo ""
echo "=== STEP 2 Complete ==="
echo "=== All Done: $(date) ==="
echo "Final model saved to: models/resnet50_best.pth"
