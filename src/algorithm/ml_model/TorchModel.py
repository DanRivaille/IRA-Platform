import numpy as np
from torch import no_grad, cuda, save, load
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from src.algorithm.ml_model.MLModel import MLModel
from src.algorithm.ml_model.models.Autoencoder import Autoencoder
from src.config.ConfigParams import ConfigParams
from src.algorithm.ml_model.History import History
from src.utils.Plotter import Plotter


class TorchModel(MLModel):

    def __init__(self, identifier: str, input_length: int, learning_rate: float):
        super().__init__(identifier)

        self.device = TorchModel.__get_device()

        self.model = Autoencoder(input_length)
        self.model.to(self.device)

        self.criterion_train = MSELoss()
        self.criterion_test = MSELoss(reduction='none')
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

    @staticmethod
    def get_file_extension():
        return '.pth'

    @staticmethod
    def __get_device():
        if cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device

    @staticmethod
    def load(config: ConfigParams, identifier: str, path: str):
        torch_model = TorchModel.create(config, identifier)
        torch_model.model.load_state_dict(load(path))
        torch_model.model.eval()
        torch_model.model.to(torch_model.device)

        return torch_model

    @staticmethod
    def create(config: ConfigParams, identifier: str):
        sequences_length = config.get_params_dict('preprocess_params')['sequences_length']
        learning_rate = config.get_params_dict('train_params')['learning_rate']
        return TorchModel(identifier, sequences_length, learning_rate)

    def save(self, config: ConfigParams, path: str):
        save(self.model.state_dict(), path)

    def train(self, config: ConfigParams, trainloader: DataLoader, validationloader: DataLoader | None) -> History:
        num_epochs = config.get_params_dict('train_params')['num_epochs']

        train_error = []
        validation_error = []
        learning_rate_updating = []

        for epoch in range(num_epochs):
            loss = self.run_epoch(trainloader)
            train_error.append(loss)

            if validationloader is not None:
                validation_loss = self.run_epoch(validationloader, is_train=False)
                validation_error.append(validation_loss)

            if (epoch % 5) == 0:
                print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss: .7f}')

            learning_rate_updating.append(config.get_params_dict('train_params')['learning_rate'])

        Plotter.plot_training_curves(train_error, validation_error,
                                     filename="/home/ivan.santos/repositories/IRA-Platform/train.png")

        train_error_per_sample = self.run_test_epoch(trainloader)
        return History(train_error, validation_error, learning_rate_updating, train_error_per_sample)

    def run_epoch(self, dataloader: DataLoader, is_train=True) -> float:
        loss = None

        for trainbatch in dataloader:
            batch = trainbatch.to(self.device)

            output = self.model(batch)
            loss = self.criterion_train(output, batch.data)

            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    # TODO: Check this function
    def test(self, config: ConfigParams, testloader: DataLoader, validationloader: DataLoader):
        pass

    def run_test_epoch(self, dataloader: DataLoader) -> np.ndarray:
        feature_vector = []

        with no_grad():
            for databatch in dataloader:
                signals = databatch.to(self.device)
                output = self.model(signals)

                current_feature_value = np.mean(self.criterion_test(output, signals.data).cpu().numpy(), axis=1)
                feature_vector.append(current_feature_value)

        return np.concatenate(feature_vector).flatten()
