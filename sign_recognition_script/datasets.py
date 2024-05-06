import os

import torch
import albumentations as A
import numpy as np

from torchvision import datasets
from torch.utils.data import DataLoader, Subset, Dataset
from albumentations.pytorch import ToTensorV2

# Required constants.
ROOT_DIR = os.path.join('sign_recognition_script','input', 'GTSRB_Final_Training_Images', 'GTSRB', 'Final_Training', 'Images')
VALID_SPLIT = 0.1
RESIZE_TO = 224  # Image size of resize when applying transforms.
BATCH_SIZE = 64
NUM_WORKERS = 4  # Number of parallel processes for data preparation.


# Training transforms.
class TrainTransforms:
    def __init__(self, resize_to):
        self.transforms = A.Compose([
            A.Resize(resize_to, resize_to),
            A.RandomBrightnessContrast(),
            A.RandomFog(),
            A.RandomRain(),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']


# Validation transforms.
class ValidTransforms:
    def __init__(self, resize_to):
        self.transforms = A.Compose([
            A.Resize(resize_to, resize_to),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']


class FilteredDataset(Dataset):
    def __init__(self, original_dataset, chosen_labels):
        self.original_dataset = original_dataset
        self.chosen_labels = set(chosen_labels)

        # Filter indices of the original dataset for the chosen labels
        self.filtered_indices = [
            i for i, (_, label) in enumerate(original_dataset)
            if label in self.chosen_labels
        ]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # Map the filtered idx back to the original dataset's idx
        original_idx = self.filtered_indices[idx]
        return self.original_dataset[original_idx]


def get_datasets(root_dir: str = ROOT_DIR, chosen_labels=None):
    """
    Function to prepare the Datasets.

    Returns the training and validation datasets along 
    with the class names.
    """
    dataset = datasets.ImageFolder(
        root_dir,
        transform=(TrainTransforms(RESIZE_TO))
    )
    dataset_test = datasets.ImageFolder(
        root_dir,
        transform=(ValidTransforms(RESIZE_TO))
    )

    # # Check if filtering by labels is requested
    # if chosen_labels is not None:
    #     # Apply filtering
    #     dataset = FilteredDataset(dataset, chosen_labels)

    dataset_size = len(dataset)

    # Calculate the validation dataset size.
    valid_size = int(VALID_SPLIT * dataset_size)
    # Radomize the data indices.
    indices = torch.randperm(len(dataset)).tolist()
    # Training and validation sets.
    dataset_train = Subset(dataset, indices[:-valid_size])
    dataset_valid = Subset(dataset_test, indices[-valid_size:])
    # dataset_train = Subset(dataset, indices[:100])
    # dataset_valid = Subset(dataset_test, indices[:1200])

    return dataset_train, dataset_valid, dataset.classes


# def get_data_loaders(dataset_train, dataset_valid):
#     """
#     Prepares the training and validation data loaders.
#
#     :param dataset_train: The training dataset.
#     :param dataset_valid: The validation dataset.
#
#     Returns the training and validation data loaders.
#     """
#     train_loader = DataLoader(
#         dataset_train, batch_size=BATCH_SIZE,
#         shuffle=True, num_workers=NUM_WORKERS
#     )
#     valid_loader = DataLoader(
#         dataset_valid, batch_size=BATCH_SIZE,
#         shuffle=False, num_workers=NUM_WORKERS
#     )
#     return train_loader, valid_loader

def get_data_loaders(dataset_train, dataset_valid, chosen_labels=None):
    """
    Prepares the training and validation data loaders.
    Optionally filters the datasets to only include specified labels.

    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    :param chosen_labels: Optional list of labels to filter datasets.

    Returns the training and validation data loaders.
    """
    if chosen_labels is not None:
        dataset_train = FilteredDataset(dataset_train, chosen_labels)
        # dataset_valid = FilteredDataset(dataset_valid, chosen_labels)

    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, valid_loader
