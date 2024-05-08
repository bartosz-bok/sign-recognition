import os

import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from src.utils import extract_epoch_number, print_pretty
from src.model_classes.local_model import LocalModel


class CentralModel:

    def __init__(self,
                 local_models_dir: str = 'models/',
                 central_model_dir: str = '',
                 scenario: str = 'static_federate',
                 device: str = ('cuda' if torch.cuda.is_available() else 'cpu'),
                 num_classes: int = 1,
                 name: str = 'central_model',
                 local_model_names: list[str] = None,
                 importance_of_local_models: str = 'equal',
                 ):
        """
        Initializes the CentralModel instance with the specified parameters.

        :param local_models_dir: Directory where local models are stored.
        :param central_model_dir: Directory where the central model will be saved.
        :param scenario: Scenario type for model aggregation. Options are 'static_federate' and 'dynamic_federate'.
        :param device: Device to use for computations. Defaults to GPU if available, else CPU.
        :param num_classes: Number of classes for classification.
        :param name: Name of the central model for identification.
        :param local_model_names: List of names for the local models.
        :param importance_of_local_models: Strategy to weigh local models during aggregation. Default is 'equal'.
        """

        if local_model_names is None:
            local_model_names = ['']

        self.model = LocalModel(
            num_classes=num_classes,
            device=device,
        ).to(device).model

        # Set params
        self.local_models_dir = local_models_dir
        self.central_model_dir = central_model_dir
        self.scenario = scenario
        self.device = device
        self.num_classes = num_classes
        self.name = name
        self.local_model_names = local_model_names
        self.importance_of_local_models = importance_of_local_models
        self.model_names_for_each_epoch = {}
        self.model_weights = None

        print(f"Chosen scenario: {self.scenario}")

    def validate(self,
                 testloader,
                 class_names: list[str]) -> (list, list):
        """
        Validate the central model using a test data loader.

        :param testloader: The DataLoader for test data.
        :param class_names: List of class names corresponding to label indices.
        :return: A tuple containing the average loss and accuracy for the validation dataset.
        """
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

    def _model_path(self) -> os.path:
        """
        Constructs the file path for saving the central model based on its name and directory.

        :return: The full file path for the model.
        """
        return os.path.join(self.central_model_dir, f'{self.name}.pth')

    def save(self) -> None:
        """
        Saves the central model's state dictionary to a file in the designated directory.
        """
        os.makedirs(self.central_model_dir, exist_ok=True)
        torch.save({'model_state_dict': self.model.state_dict()}, self._model_path())
        print(f"[INFO] Model {self.name}.pth saved!")

    def load_central_model(self) -> None:
        """
        Loads the central model's state dictionary from a file.
        """
        model_path = self._model_path()
        print(f"[INFO] Loading model {self.name}.pth from '{self.central_model_dir}'...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found at '{model_path}'")
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Model {self.name}.pth loaded successfully!")

    def update_central_model(self, epoch: int) -> None:
        """
        Updates the central model by aggregating weights from local models according to the specified scenario.

        :param epoch: Current epoch number to load specific model snapshots.
        """
        if self.scenario == 'static_federate':
            self.update_central_model_based_on_static_federate_scenario(epoch=epoch)
        elif self.scenario == 'dynamic_federate':
            pass
        else:
            raise ValueError(f'Scenario {self.scenario} is not supported!')

    def update_central_model_based_on_static_federate_scenario(self, epoch: int) -> None:
        """
        Updates the central model based on a static federated scenario using the specific epoch.

        :param epoch: Epoch number to load specific local model snapshots for aggregation.
        """
        self.load_local_models_from_specific_epoch(epoch=epoch)
        # Get the smallest max epoch from models
        max_epochs = min(max(extract_epoch_number(file) for file in files)
                         for files in self.model_names_for_each_epoch.values())
        print(f'The smallest max epoch number in given models: {max_epochs}.')

        tmp_local_model = LocalModel(num_classes=self.num_classes, device=self.device).to(self.device)

        aggregated_weights = None
        for model_idx, model_name in enumerate(self.local_model_names):
            model_path = os.path.join(self.local_models_dir, model_name, f'{model_name}_epoch_{epoch}.pth')
            if os.path.exists(model_path):
                network_state_dict = torch.load(model_path, map_location=torch.device(self.device))
                tmp_local_model.model.load_state_dict(network_state_dict)

                if aggregated_weights is None:
                    aggregated_weights = {key: val.clone() * self.model_weights[model_idx] for key, val in
                                          tmp_local_model.model.state_dict().items()}
                else:
                    for key, val in tmp_local_model.model.state_dict().items():
                        aggregated_weights[key] += val * self.model_weights[model_idx]
            else:
                raise ValueError(f"Given model path ({model_path}) doesn't exist!")

        # Update of central network weights
        if aggregated_weights is not None:
            self.model.load_state_dict(aggregated_weights)

    def load_local_model(self, version: int, epoch: int = None) -> LocalModel:
        """
        Loads a specific version of a local model, optionally at a specified epoch.

        :param version: Version number of the local model to load.
        :param epoch: Epoch number to load the model from. Defaults to the latest available epoch.
        :return: The loaded local model.
        """
        local_model_name = f'model_v{version}'
        model_dir = os.path.join(self.local_models_dir, local_model_name)

        if epoch is None:
            model_files = [f for f in os.listdir(model_dir) if
                           f.startswith(f'{local_model_name}_epoch_') and f.endswith('.pth')]
            epoch = max(extract_epoch_number(f) for f in model_files)
            print(f"[INFO] No epoch provided, loading latest epoch: {epoch}")

        model_path = os.path.join(model_dir, f"{local_model_name}_epoch_{epoch}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found at '{model_path}'")

        print(f"[INFO] Loading model '{local_model_name}' from '{self.local_models_dir}'...")
        checkpoint = torch.load(model_path)
        local_model = LocalModel(num_classes=self.num_classes, device=self.device).to(self.device)
        local_model.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Model {local_model_name} loaded successfully!")

        return local_model

    def load_local_models_from_specific_epoch(self, epoch: int) -> None:
        """
        Loads all local models from a specific epoch.

        :param epoch: Epoch number from which to load local models.
        """
        self.update_model_names_for_each_epoch_dict()

        epoch_suffix = f'_epoch_{epoch}.pth'
        self.local_model_names = [
            model_name
            for model_list in self.model_names_for_each_epoch.values()
            for model_name in model_list
            if model_name.endswith(epoch_suffix)
        ]

    def update_model_names_for_each_epoch_dict(self) -> None:
        """
        Updates the dictionary that maps local model names to their available epochs based on the current local model names list.
        """
        self.model_names_for_each_epoch = {}
        for local_model_name in self.local_model_names:
            single_model_path = os.path.join(self.local_models_dir, local_model_name)
            files_name = os.listdir(single_model_path)

            self.model_names_for_each_epoch[local_model_name] = [
                file for file in files_name if file.startswith(f'{local_model_name}_epoch_') and file.endswith('.pth')
            ]
        print_pretty(models_dict=self.model_names_for_each_epoch)

    def get_importance_of_models(self) -> None:
        """
        Determines the importance weights for local models based on the specified importance strategy.
        """
        if self.importance_of_local_models == 'equal':
            # Every local model has same weight
            self.model_weights = [1 / len(self.local_model_names) for _ in range(len(self.local_model_names))]
        else:
            raise ValueError(f'{self.importance_of_local_models} importance is not supported!')
