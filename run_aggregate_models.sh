#!/bin/bash

# Load configurations
source config.sh

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Running the aggregate models script
echo "[INFO][sh] Aggregate models script started..."
python3 src/aggregate_models.py --local-models-dir $MODELS_DIR --central-model-dir $CENTRAL_MODEL_DIR --scenario $SCENARIO
