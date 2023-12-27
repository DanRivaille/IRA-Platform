import numpy as np
import time
from torch import no_grad, cuda, save, load, enable_grad
from torch.nn import MSELoss, Module
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.algorithm.ml_model.History import History
from src.algorithm.ml_model.MLModel import MLModel
from src.algorithm.ml_model.ModelType import ModelType
from src.algorithm.ml_model.models.Autoencoder import Autoencoder
from src.config.ConfigParams import ConfigParams
from src.utils.Plotter import Plotter


class TorchModel(MLModel):
    """
    A class representing a Torch model.
    """
    def __init__(self, identifier: str, input_length: int, learning_rate: float):
        """
        Initializes an instance of TorchModel.
        @param identifier An identifier for the model.
        @param input_length The size of each input and output layer entry.
        @param learning_rate The learning rate for the model.
        """
        super().__init__(identifier)

        self.device = TorchModel.__get_device()

        self.model: Module = Autoencoder(input_length)
        self.model.to(self.device)

        self.criterion = MSELoss
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

    @staticmethod
    def get_file_extension():
        """
        Get the file extension for the file with the parameters of the Torch model.
        """
        return '.pth'

    @staticmethod
    def get_model_type() -> ModelType:
        """
        Get the Torch model type.
        """
        return ModelType.TORCH_MODEL

    @staticmethod
    def __get_device():
        """
        Gets the device string for PyTorch operations, either CUDA (GPU) or CPU.
        """
        if cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device

    @staticmethod
    def load(config: ConfigParams, identifier: str, path: str):
        """
        Loads the Torch model with the specified configurations.
        @param config A class for loading configuration parameters related to a model.
        @param identifier An identifier for the model.
        @param path The directory path of the configurations to load.
        """
        torch_model = TorchModel.create(config, identifier)
        torch_model.model.load_state_dict(load(path))
        torch_model.model.eval()
        torch_model.model.to(torch_model.device)

        return torch_model

    @staticmethod
    def create(config: ConfigParams, identifier: str):
        """
        Creates the Torch model with the specified configurations.
        @param config A class for loading configuration parameters related to a model.
        @param identifier An identifier for the model.
        """
        sequences_length = config.get_params_dict('train_params')['sequences_length']
        learning_rate = config.get_params_dict('train_params')['learning_rate']
        return TorchModel(identifier, sequences_length, learning_rate)

    def save(self, config: ConfigParams, path: str):
        """
        Saves the Torch model to the specified path.
        @param path The directory path of the configurations to save.
        """
        save(self.model.state_dict(), path)

    def train(self, config: ConfigParams, trainloader: DataLoader, validationloader: DataLoader | None) -> History:
        """
        Runs the train process of the Torch model.
        @param config A class for loading configuration parameters related to a model.
        @param trainloader DataLoader containing training data.
        @param validationloader DataLoader containing validation data.
        @return History A class for saving the training history data of model execution.
        """
        num_epochs = config.get_params_dict('train_params')['num_epochs']

        train_error = []
        validation_error = []
        learning_rate_updating = []
        start_time = time.time()

        for epoch in range(num_epochs):
            _, loss = self.predict(trainloader, is_train_data=True, criterion_reduction='mean')
            train_error.append(loss)

            if validationloader is not None:
                _, validation_loss = self.predict(trainloader, is_train_data=False, criterion_reduction='mean')
                validation_error.append(validation_loss)

            if (epoch % 5) == 0:
                print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss: .7f}')

            learning_rate_updating.append(config.get_params_dict('train_params')['learning_rate'])

        end_time = time.time()
        elapsed_time = end_time - start_time

        #Plotter.plot_training_curves(train_error, validation_error,
        #                             filename="/home/ivan.santos/repositories/IRA-Platform/train.png")

        _, train_error_per_sample = self.predict(trainloader, is_train_data=False, criterion_reduction='none')
        return History(train_error, validation_error, learning_rate_updating, train_error_per_sample, elapsed_time)

    # TODO: Check this function
    def test(self, config: ConfigParams, testloader: DataLoader, validationloader: DataLoader):
        pass

    
    def predict(self, dataloader: DataLoader, is_train_data: bool = False, criterion_reduction: str = 'mean') -> tuple:
        """
        Predict the data in dataloader using the current state of the model.
        @param is_train_data If it is True, process the data with the gradient enable and compute one step of the optimizer
        @param criterion_reduction ('mean', 'none'). 'mean' return the loss mean of the batch. 'none' return the loss for
        each sample.
        @return Tuple: (predictions, errors)
        """
        losses = []
        criterion = self.criterion(reduction=criterion_reduction)

        gradient = enable_grad if is_train_data else no_grad

        with gradient():
            for databatch in dataloader:
                batch = databatch.to(self.device)
                output = self.model(batch)

                if 'none' == criterion_reduction:
                    current_batch_loss = np.mean(criterion(output, batch.data).cpu().numpy(), axis=1)
                else:
                    current_batch_loss = criterion(output, batch.data)

                losses.append(current_batch_loss)

                if is_train_data:
                    self.optimizer.zero_grad()
                    losses[-1].backward()
                    self.optimizer.step()

        if 'none' == criterion_reduction:
            return None, np.concatenate(losses).flatten()
        else:
            return None, losses[-1].item()
