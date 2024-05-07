import os
import argparse

import torch
import torch.nn as nn

from federated_scripts.local_model import LocalModel
from federated_scripts.utils import save_plots
from federated_scripts.datasets import get_datasets, get_data_loaders
from utils import extract_epoch_number, print_pretty
from federated_scripts.config import config_params

torch.manual_seed(2)

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', type=int, default=10,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    '-lm', '--n-local-models', dest='n_local_models', type=int, default=2,
    help='Number of local models'
)
parser.add_argument(
    '-lr', '--learning-rate', type=float,
    dest='learning_rate', default=0.001,
    help='Learning rate for training the model'
)
parser.add_argument(
    '-pw', '--pretrained', action='store_true',
    help='whether to use pretrained weihgts or not'
)
parser.add_argument(
    '-ft', '--fine-tune', dest='fine_tune', action='store_true',
    help='whether to train all layers or not'
)
parser.add_argument(
    '-us', '--use-scheduler', dest='use_scheduler', action='store_true',
    help='whether to use scheduler'
)

parser_used = True

labels_set = {
    1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    2: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    3: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    4: [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    5: [40, 41, 42]
}

if __name__ == '__main__':

    if parser_used:
        args = vars(parser.parse_args())
    else:
        args = config_params

    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets()
    print(f"[INFO].py: Number of training images: {len(dataset_train)}")
    print(f"[INFO].py: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO].py: Class names: {dataset_classes}\n")

    # Learning_parameters.
    pretrained = args['pretrained'],
    fine_tune = args['fine_tune'],
    lr = args['learning_rate']
    epochs = args['epochs']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # Define local models
    number_of_local_models = args['n_local_models']
    local_models = {}
    for i in range(number_of_local_models):
        local_models[f'model_v{i + 1}'] = LocalModel(
            pretrained=pretrained[0],
            fine_tune=fine_tune[0],
            num_classes=len(dataset_classes),
            lr=lr,
            device=device,
            version=i + 1,
        )
        local_models[f'model_v{i + 1}'].to(device)
    model_names = [f'model_v{i + 1}' for i in range(number_of_local_models)]

    # Train local models
    for i, local_model_name in enumerate(local_models.keys()):
        local_model = local_models[local_model_name]
        version = i + 1
        print('-' * 50)
        print(f'Training local model: {local_model.name} started...')

        # Load the training and validation data loaders.
        train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid, chosen_labels=labels_set[version])

        # Define scheduler
        if args['use_scheduler']:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                local_model.optimizer,
                T_0=10,
                T_mult=1,
                verbose=True
            )
        else:
            scheduler = None

        # Lists to keep track of losses and accuracies.
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []
        # Start the training.
        for epoch in range(epochs):
            print(f"[INFO].py: Epoch {epoch + 1} of {epochs}")

            train_epoch_loss, train_epoch_acc = local_model.train_one_epoch(
                trainloader=train_loader,
                scheduler=scheduler
            )
            valid_epoch_loss, valid_epoch_acc = local_model.validate(
                testloader=valid_loader,
                class_names=dataset_classes, )
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)
            print(f"Training loss: {round(train_epoch_loss, 3)}, training acc: {round(train_epoch_acc, 3)}")
            print(f"Validation loss: {round(valid_epoch_loss, 3)}, validation acc: {round(valid_epoch_acc, 3)}")
            print('-' * 20)

            # Save the trained model weights.
            local_model.save()

        # Save the loss and accuracy plots.
        save_plots(
            train_acc=train_acc,
            valid_acc=valid_acc,
            train_loss=train_loss,
            valid_loss=valid_loss,
            version=version
        )
        print(f'Training local model: {local_model.name} completed...\n')

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
    model_weights = [1 / len(model_names) for _ in range(len(model_names))]

    # Get the smallest max epoch from models
    max_epochs = min_max_epoch = min(max(extract_epoch_number(file) for file in files)
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
