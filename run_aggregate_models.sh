#!/bin/bash

# Load configurations
source config.sh

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Running the train model script
echo "[INFO]: Train model script started..."
python3 federated_scripts/aggregate_models.py --local-models-dir $MODELS_DIR --central-model-dir $CENTRAL_MODEL_DIR --scenario $SCENARIO --model-names $MODEL_NAMES
