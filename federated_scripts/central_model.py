import os

import torch

from src.model import build_model
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
        single_model_path = os.path.join(models_path, single_model_name)
        files_name = os.listdir(single_model_path)

        model_names_for_each_epoch[single_model_name] = [
            file for file in files_name if file.startswith(f'{single_model_name}_epoch_') and file.endswith('.m.pth')
        ]
    print_pretty(models_dict=model_names_for_each_epoch)

    # Get model weights (every local model has same weight)
    model_weights = [1 / len(model_names) for _ in range(len(model_names))]

    # Get the smallest max epoch from models
    max_epochs = min_max_epoch = min(max(extract_epoch_number(file) for file in files)
                                     for files in model_names_for_each_epoch.values())
    print(f'The smallest max epoch number in given models: {max_epochs}.')

    # Prepare DataLoader to test data
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=True)

    # Get weights to central network
    test_losses = []
    for epoch in range(1, max_epochs + 1):
        aggregated_weights = None
        for model_idx, model_name in enumerate(model_names):
            model_path = os.path.join(models_path, model_name, f'{model_name}_epoch_{epoch}.m.pth')
            if os.path.exists(model_path):
                network_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                local_network.load_state_dict(network_state_dict)

                if aggregated_weights is None:
                    aggregated_weights = {key: val.clone() * model_weights[model_idx] for key, val in local_network.state_dict().items()}
                else:
                    for key, val in local_network.state_dict().items():
                        aggregated_weights[key] += val * model_weights[model_idx]
            else:
                raise ValueError(f"Given model path ({model_path}) doesn't exist!")

        # Update of central network weights
        if aggregated_weights is not None:
            central_network.load_state_dict(aggregated_weights)

        # Test central network
        test(model=central_network, test_loader=test_loader, test_losses=test_losses)
