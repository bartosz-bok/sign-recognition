#!/bin/bash

# Load configurations
source config.sh

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Running the train model script
echo "[INFO][sh] Train model script started..."
python3 src/train_model.py --version $VERSION --epochs $EPOCHS --learning-rate $LEARNING_RATE --pretrained --fine-tune --models-dir $MODELS_DIR
