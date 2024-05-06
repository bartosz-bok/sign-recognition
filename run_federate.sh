#!/bin/bash

# Load configurations
source config.sh

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:~/github_projects/sign-recognition/sign-recognition"

# Define the number of label sets
num_versions=5

# Training models with different label sets
for version in $(seq 1 $num_versions)
do
  echo "[INFO].sh: Training model with label set $version"
  python3 sign_recognition_script/train.py --pretrained --fine-tune --epochs $EPOCHS --learning-rate $LEARNING_RATE --version $version --labels-set $version
done

# Running the central model script
echo "[INFO].sh: Aggregating models into a central model"
python3 federated_scripts/central_model.py
