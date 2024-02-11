import os

import torch
import torch.nn as nn

from src.model import build_model
from src.datasets import get_datasets, get_data_loaders
from src.utils import validate
from utils import extract_epoch_number, print_pretty
from params import models_path, BATCH_SIZE_TEST

torch.manual_seed(2)

model_names = ['model_v1', 'model_v2', 'model_v3']

device = 'cpu'

if __name__ == '__main__':
    # Prepare dict to get model names with their epoch versions
    model_names_for_each_epoch = {}
    # Prepare central model
    central_network = build_model(
        pretrained=False,
        fine_tune=False,
        num_classes=43
    ).to(device)
    # Model to load local models parameters
    local_network = build_model(
        pretrained=False,
        fine_tune=False,
        num_classes=43
    ).to(device)

    # Get model names with their epochs versions
    for single_model_name in model_names:
        single_model_path = os.path.join('..', 'models', single_model_name)
        files_name = os.listdir(single_model_path)

        model_names_for_each_epoch[single_model_name] = [
            file for file in files_name if file.startswith(f'{single_model_name}_epoch_') and file.endswith('.pth')
        ]
    print_pretty(models_dict=model_names_for_each_epoch)

    # Get model weights (every local model has same weight)
    model_weights = [1 / len(model_names) for _ in range(len(model_names))]

    # Get the smallest max epoch from models
    max_epochs = min_max_epoch = min(max(extract_epoch_number(file) for file in files)
                                     for files in model_names_for_each_epoch.values())
    print(f'The smallest max epoch number in given models: {max_epochs}.')

    # Get weights to central network
    test_losses = []
    for epoch in range(1, max_epochs + 1):
        aggregated_weights = None
        for model_idx, model_name in enumerate(model_names):
            model_path = os.path.join('..', 'models', model_name, f'{model_name}_epoch_{epoch}.pth')
            if os.path.exists(model_path):
                network_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                local_network.load_state_dict(network_state_dict)

                if aggregated_weights is None:
                    aggregated_weights = {key: val.clone() * model_weights[model_idx] for key, val in
                                          local_network.state_dict().items()}
                else:
                    for key, val in local_network.state_dict().items():
                        aggregated_weights[key] += val * model_weights[model_idx]
            else:
                raise ValueError(f"Given model path ({model_path}) doesn't exist!")

        # Update of central network weights
        if aggregated_weights is not None:
            central_network.load_state_dict(aggregated_weights)

        # Load the training and validation datasets.
        dataset_train, dataset_valid, dataset_classes = get_datasets(root_dir=os.path.join('..',
                                                                                           'sign_recognition_script',
                                                                                           'input',
                                                                                           'GTSRB_Final_Training_Images',
                                                                                           'GTSRB',
                                                                                           'Final_Training',
                                                                                           'Images')
                                                                     )

        # Load the training and validation data loaders.
        _, valid_loader = get_data_loaders(dataset_train, dataset_valid)

        # Loss function.
        criterion = nn.CrossEntropyLoss()

        # Test central network
        _, valid_epoch_acc = validate(model=central_network,
                                      testloader=valid_loader,
                                      criterion=criterion,
                                      class_names=dataset_classes,
                                      device=device)
