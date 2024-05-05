import torch
import argparse
import torch.nn as nn
import torch.optim as optim

from src.model import build_model
from src.datasets import get_datasets, get_data_loaders
from config import config_params
from utils import save_model, save_plots, train

from src.utils import validate

seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', type=int, default=10,
    help='Number of epochs to train our network for'
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
    '--version', type=int,
    dest='version', default=1,
    help='Version of the model'
)
parser.add_argument(
    '-us', '--use-scheduler', dest='use_scheduler', action='store_true',
    help='whether to use scheduler'
)

parser_used = False

# chosen_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# chosen_labels = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# chosen_labels = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
# chosen_labels = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
chosen_labels = [40, 41, 42]

if __name__ == '__main__':

    if parser_used:
        args = vars(parser.parse_args())
    else:
        args = config_params

    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets()
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Class names: {dataset_classes}\n")
    # Load the training and validation data loaders.
    train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid, chosen_labels=chosen_labels)

    # Learning_parameters. 
    lr = args['learning_rate']
    epochs = args['epochs']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")

    # Load the model.
    model = build_model(
        pretrained=args['pretrained'],
        fine_tune=args['fine_tune'],
        num_classes=len(dataset_classes)
    ).to(device)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Loss function.
    criterion = nn.CrossEntropyLoss()

    # Define scheduler
    if args['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
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
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model=model,
            trainloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            epoch=epoch
        )
        valid_epoch_loss, valid_epoch_acc = validate(model=model,
                                                     testloader=valid_loader,
                                                     criterion=criterion,
                                                     class_names=dataset_classes,
                                                     device=device)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {round(train_epoch_loss, 3)}, training acc: {round(train_epoch_acc, 3)}")
        print(f"Validation loss: {round(valid_epoch_loss, 3)}, validation acc: {round(valid_epoch_acc, 3)}")
        print('-' * 50)

        # Save the trained model weights.
        save_model(model=model, version=args['version'], epoch=epoch)
    # Save the loss and accuracy plots.
    save_plots(
        train_acc=train_acc,
        valid_acc=valid_acc,
        train_loss=train_loss,
        valid_loss=valid_loss,
        version=args['version']
    )
    print('TRAINING COMPLETE')
