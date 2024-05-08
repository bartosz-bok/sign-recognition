import os
import re

import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')


def extract_epoch_number(filename):
    """
    Extracts the epoch number from a model file name.

    :param filename: The name of the model file from which the epoch number is to be extracted.
    :return: The extracted epoch number from the file name, or 0 if the epoch number cannot be found.
    """
    match = re.search(r'epoch_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


#
def print_pretty(models_dict) -> None:
    """
    Formatting and displaying the dictionary in a more readable way.

    :param models_dict: model dict to print
    """
    for model_name, files in models_dict.items():
        print(f"{model_name}:")
        files.sort()
        for file in files:
            print(f"  - {file}")
    print('\n')


def save_plots(train_acc, valid_acc, train_loss, valid_loss, version: int):
    """
    Function to save the loss and accuracy plots to disk.
    """
    model_name = f'model_v{version}'
    model_path = os.path.join('models', model_name)
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{model_path}/accuracy.png")

    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(f"{model_path}/loss.png")
