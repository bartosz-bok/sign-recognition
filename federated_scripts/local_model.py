import os

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights

import matplotlib
import matplotlib.pyplot as plt

from tqdm.auto import tqdm


class LocalModel:

    def __init__(self,
                 pretrained: bool = True,
                 fine_tune: bool = False,
                 num_classes: int = 10,
                 lr: float = 0.001,
                 epoch: int = 0,
                 optimizer: optim = optim.Adam,
                 criterion=nn.CrossEntropyLoss(),
                 device: str = ('cuda' if torch.cuda.is_available() else 'cpu'),
                 version: int = 0,
                 ):
        if pretrained:
            print('[INFO].py: Loading pre-trained weights')
            weights = MobileNet_V3_Large_Weights.DEFAULT
        else:
            print('[INFO].py: Not loading pre-trained weights')
            weights = None

        self.model = models.mobilenet_v3_large(weights=weights)

        if fine_tune:
            print('[INFO].py: Fine-tuning all layers...')
            for params in self.model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print('[INFO].py: Freezing hidden layers...')
            for params in self.model.parameters():
                params.requires_grad = False

        # Change the final classification head.
        self.model.classifier[3] = nn.Linear(in_features=1280, out_features=num_classes)

        # Set model hiper-params
        self.lr = lr
        self.epoch = epoch
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr)
        self.criterion = criterion
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
                        trainloader,
                        scheduler=None
                        ):
        self.model.train()
        print(f'Model training: epoch {self.epoch}')
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

            if scheduler is not None:
                scheduler.step(self.epoch + i / iters)

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

    def save(self) -> None:
        """
        Function to save the trained model to disk.
        """
        model_path = os.path.join('models', self.name)
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_path, f"{self.name}_epoch_{self.epoch}.pth"))
        print(f"model {self.name}_epoch_{self.epoch}.pth saved!")
