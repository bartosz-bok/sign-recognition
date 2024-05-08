import os

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights

import matplotlib
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from src.utils import extract_epoch_number


class LocalModel:

    def __init__(self,
                 pretrained: bool = True,
                 fine_tune: bool = False,
                 num_classes: int = 10,
                 lr: float = 0.001,
                 epoch: int = 0,
                 optimizer: optim = optim.Adam,
                 criterion=nn.CrossEntropyLoss(),
                 scheduler: torch.optim.lr_scheduler = None,
                 device: str = ('cuda' if torch.cuda.is_available() else 'cpu'),
                 version: int = 0,
                 ):
        if pretrained:
            print('[INFO] Loading pre-trained weights')
            weights = MobileNet_V3_Large_Weights.DEFAULT
        else:
            print('[INFO] Not loading pre-trained weights')
            weights = None

        self.model = models.mobilenet_v3_large(weights=weights)

        if fine_tune:
            print('[INFO] Fine-tuning all layers...')
            for params in self.model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print('[INFO] Freezing hidden layers...')
            for params in self.model.parameters():
                params.requires_grad = False

        # Change the final classification head.
        self.model.classifier[3] = nn.Linear(in_features=1280, out_features=num_classes)

        # Set model hiperparams
        self.lr = lr
        self.epoch = epoch
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr)
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.version = version
        self.name = f'model_v{self.version}'

        print(f"Model {self.name} defined.")
        print(f"Computation device: {self.device}")
        print(f"Learning rate: {self.lr}")
        # print(f"Optimizer: {self.optimizer}")
        print(f"Loss criterion: {self.criterion}")

    def to(self, device):
        self.model.to(device)

    def train_one_epoch(self,
                        trainloader
                        ):
        self.model.train()
        print(f'Model training: epoch {self.epoch + 1}')
        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0
        iters = len(trainloader)
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            counter += 1
            image, labels = data
            image = image.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            # Forward pass.
            outputs = self.model(image)
            # Calculate the loss.
            loss = self.criterion(outputs, labels)
            train_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == labels).sum().item()
            # Backpropagation.
            loss.backward()
            # Update the weights.
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step(self.epoch + i / iters)

        self.epoch += 1

        # Loss and accuracy for the complete epoch.
        epoch_loss = train_running_loss / counter
        epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
        return epoch_loss, epoch_acc

    def validate(self,
                 testloader,
                 class_names):
        self.model.eval()
        print('Model validation')
        valid_running_loss = 0.0
        valid_running_correct = 0
        counter = 0

        # We need two lists to keep track of class-wise accuracy.
        class_correct = list(0. for i in range(len(class_names)))
        class_total = list(0. for i in range(len(class_names)))

        with torch.no_grad():
            for i, data in tqdm(enumerate(testloader), total=len(testloader)):
                counter += 1

                image, labels = data
                image = image.to(self.device)
                labels = labels.to(self.device)
                # Forward pass.
                outputs = self.model(image)
                # Calculate the loss.
                loss = self.criterion(outputs, labels)
                valid_running_loss += loss.item()
                # Calculate the accuracy.
                _, preds = torch.max(outputs.data, 1)
                valid_running_correct += (preds == labels).sum().item()

                # Calculate the accuracy for each class.
                correct = (preds == labels).squeeze()
                for j in range(len(preds)):
                    label = labels[j]
                    class_correct[label] += correct[j].item()
                    class_total[label] += 1

        # Loss and accuracy for the complete epoch.
        epoch_loss = valid_running_loss / counter
        epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))

        # Print the accuracy for each class after every epoch.
        print('\n')
        for i in range(len(class_names)):
            print(f"Accuracy of class {class_names[i]}: {100 * class_correct[i] / class_total[i]}")
        print('\n')
        return epoch_loss, epoch_acc

    def save(self, models_dir: str = 'models') -> None:
        """
        Function to save the trained model to disk.
        """
        model_path = os.path.join(models_dir, self.name)
        os.makedirs(model_path, exist_ok=True)
        model_name_with_epoch = f"{self.name}_epoch_{self.epoch}.pth"

        artifacts_to_save = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None
        }

        torch.save(artifacts_to_save, os.path.join(model_path, model_name_with_epoch))

        print(f"[INFO] Model {model_name_with_epoch} saved!")

    def load(self, epoch: int = None, models_dir: str = 'models') -> None:
        """
        Function to load a trained model from disk.
        """
        model_dir = os.path.join(models_dir, self.name)
        print(f"[INFO] Loading model for '{self.name}' from '{model_dir}'...")

        if epoch is None:
            # Automatically determine the latest epoch model to load.
            model_files = [f for f in os.listdir(model_dir) if
                           f.startswith(f'{self.name}_epoch_') and f.endswith('.pth')]
            epoch = max(extract_epoch_number(f) for f in model_files)
            print(f"[INFO] No epoch provided, loading latest epoch: {epoch}")

        model_path = os.path.join(model_dir, f"{self.name}_epoch_{epoch}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found at '{model_path}'")

        # Load the model checkpoint.
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = epoch
        print(f"[INFO] Model loaded successfully: {self.name}_epoch_{self.epoch}")

