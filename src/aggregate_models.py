import os
import argparse

import torch
import torch.nn as nn

from src.model_classes.local_model import LocalModel
from src.model_classes.central_model import CentralModel
from src.datasets import get_datasets, get_data_loaders
from utils import extract_epoch_number, print_pretty
from src.config import aggregate_models_config_params

torch.manual_seed(2)

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-lmd', '--local-models-dir', type=str,
    dest='local_models_dir', default='models/',
    help='directory of local models to load'
)
parser.add_argument(
    '-cmd', '--central-model-dir', type=str,
    dest='central_model_dir', default='central_models/',
    help='directory of central model to save'
)
parser.add_argument(
    '-s', '--scenario', type=str,
    dest='scenario', default='static_aggregate',
    help='scenario of federated learning'
)
parser.add_argument(
    '-mn', '--model-names', dest='model_names',
    nargs='+', default=[],
    help='List of model names'
)

parser_used = False

if __name__ == '__main__':

    if parser_used:
        args = vars(parser.parse_args())
    else:
        args = aggregate_models_config_params

    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets()
    print(f"[INFO].py: Number of training images: {len(dataset_train)}")
    print(f"[INFO].py: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO].py: Class names: {dataset_classes}\n")

    # Load parameters.
    local_models_dir = args['local_models_dir'],
    central_model_dir = args['central_model_dir'],
    scenario = args['scenario']
    model_names = args['model_names']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    central_model = CentralModel(
        central_model_dir='central_models/',
        num_classes=len(dataset_classes),
        name='central_model_v1',
        scenario='static_federate',
    )

    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets(root_dir=os.path.join('dataset',
                                                                                       'input',
                                                                                       'GTSRB_Final_Training_Images',
                                                                                       'GTSRB',
                                                                                       'Final_Training',
                                                                                       'Images'))

    max_epochs = 3
    for epoch in range(1, max_epochs + 1):
        central_model.update_central_model(epoch=epoch)

        # Load the training and validation data loaders.
        _, valid_loader = get_data_loaders(dataset_train, dataset_valid)

        # Test central network
        _, valid_epoch_acc = central_model.validate(
            testloader=valid_loader,
            class_names=dataset_classes,
        )
