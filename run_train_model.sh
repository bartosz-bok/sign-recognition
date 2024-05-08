#!/bin/bash

# Load configurations
source config.sh

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Running the train model script
echo "[INFO]: Train model script started..."
python3 federated_scripts/train_model.py --version $VERSION --epochs $EPOCHS --learning-rate $LEARNING_RATE --pretrained --fine-tune --models-dir $MODELS_DIR
