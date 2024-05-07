#!/bin/bash

# Load configurations
source config.sh

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:~/github_projects/sign-recognition/sign-recognition"

# Running the central model script
echo "[INFO]: Federated scenario: central model finally scenario"
python3 federated_scripts/central_model_finally_scenario.py --pretrained --fine-tune --use-scheduler --epochs $EPOCHS --learning-rate $LEARNING_RATE --n-local-models $N_LOCAL_MODELS
