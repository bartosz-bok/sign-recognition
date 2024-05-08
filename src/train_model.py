import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from src.model_classes.local_model import LocalModel
from src.utils import save_plots, extract_epoch_number, print_pretty
from src.datasets import get_datasets, get_data_loaders
from src.config import train_model_config_params

torch.manual_seed(2)

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-v', '--version', type=int, default=1,
    help='version of trained model'
)
parser.add_argument(
    '-e', '--epochs', type=int, default=1,
    help='number of epochs to train our network for'
)
parser.add_argument(
    '-lr', '--learning-rate', type=float,
    dest='learning_rate', default=0.001,
    help='learning rate for training the model'
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
parser.add_argument(
    '-md', '--models-dir', type=str,
    dest='models_dir', default='models/',
    help='directory to save/load models'
)
parser.add_argument(
    '-lm', '--load-model', dest='load_model', action='store_true',
    help='whether to load model'
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
        args = train_model_config_params

    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets()
    print(f"[INFO] Number of training images: {len(dataset_train)}")
    print(f"[INFO] Number of validation images: {len(dataset_valid)}")
    print(f"[INFO] Class names: {dataset_classes}\n")

    # Learning parameters.
    pretrained = args['pretrained']
    fine_tune = args['fine_tune']
    lr = args['learning_rate']
    epochs = args['epochs']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # Experiment parameters
    model_version = args['version']
    models_dir = args['models_dir']
    load_model = args['load_model']

    # Define local model
    local_model = LocalModel(
        pretrained=pretrained,
        fine_tune=fine_tune,
        num_classes=len(dataset_classes),
        lr=lr,
        optimizer=optim.Adam,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        version=model_version,
    )
    local_model.to(device)

    # Define scheduler
    if args['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            local_model.optimizer,
            T_0=10,
            T_mult=1,
            verbose=True
        )
        local_model.scheduler = scheduler

    # If load_model than load model with the highest epoch
    if load_model:
        local_model.load(models_dir=models_dir)

    print(f'[INFO] Loading DataLoaders...')
    # Load the training and validation data loaders.
    train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid, chosen_labels=labels_set[model_version])
    print(f'[INFO] DataLoaders loaded!')

    # Train local model
    print('-' * 50)
    print(f'[INFO] Training local model: {local_model.name} started...')

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    last_epoch = local_model.epoch
    for epoch in range(last_epoch, last_epoch + epochs):
        print(f"[INFO] Epoch {epoch + 1} of {last_epoch + epochs}")

        train_epoch_loss, train_epoch_acc = local_model.train_one_epoch(
            trainloader=train_loader
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
        local_model.save(models_dir=models_dir)

    # Save the loss and accuracy plots.
    save_plots(
        train_acc=train_acc,
        valid_acc=valid_acc,
        train_loss=train_loss,
        valid_loss=valid_loss,
        version=model_version
    )
    print(f'[INFO] Training local model: {local_model.name} completed...\n')
