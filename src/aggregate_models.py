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

parser_used = True

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

    # Get model names with their epochs versions
    model_names_for_each_epoch = {}
    for single_model_name in model_names:
        single_model_path = os.path.join('models', single_model_name)
        files_name = os.listdir(single_model_path)

        model_names_for_each_epoch[single_model_name] = [
            file for file in files_name if file.startswith(f'{single_model_name}_epoch_') and file.endswith('.pth')
        ]
    print_pretty(models_dict=model_names_for_each_epoch)

    # Get model weights (every local model has same weight)
    model_weights = [1 / len(model_names) for _ in range(len(model_names))]  # zmienic na rozne mozliwosci

    # Get the smallest max epoch from models
    max_epochs = min(max(extract_epoch_number(file) for file in files)
                     for files in model_names_for_each_epoch.values())
    print(f'The smallest max epoch number in given models: {max_epochs}.')

    # Define central model
    central_model = LocalModel(
        num_classes=len(dataset_classes),
        criterion=nn.CrossEntropyLoss()
    )
    central_model.to(device)
    # Model to load local models parameters
    tmp_local_model = LocalModel(
        num_classes=len(dataset_classes),
    )
    tmp_local_model.to(device)
    # Get weights to central network
    test_losses = []
    for epoch in range(1, max_epochs + 1):
        aggregated_weights = None
        for model_idx, model_name in enumerate(model_names):
            model_path = os.path.join('models', model_name, f'{model_name}_epoch_{epoch}.pth')
            if os.path.exists(model_path):
                network_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                tmp_local_model.model.load_state_dict(network_state_dict)

                if aggregated_weights is None:
                    aggregated_weights = {key: val.clone() * model_weights[model_idx] for key, val in
                                          tmp_local_model.model.state_dict().items()}
                else:
                    for key, val in tmp_local_model.model.state_dict().items():
                        aggregated_weights[key] += val * model_weights[model_idx]
            else:
                raise ValueError(f"Given model path ({model_path}) doesn't exist!")

        # Update of central network weights
        if aggregated_weights is not None:
            central_model.model.load_state_dict(aggregated_weights)

        # Load the training and validation datasets.
        dataset_train, dataset_valid, dataset_classes = get_datasets(root_dir=os.path.join('dataset',
                                                                                           'input',
                                                                                           'GTSRB_Final_Training_Images',
                                                                                           'GTSRB',
                                                                                           'Final_Training',
                                                                                           'Images')
                                                                     )

        # Load the training and validation data loaders.
        _, valid_loader = get_data_loaders(dataset_train, dataset_valid)

        # Test central network
        _, valid_epoch_acc = central_model.validate(
            testloader=valid_loader,
            class_names=dataset_classes,
        )
