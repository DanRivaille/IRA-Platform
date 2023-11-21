from torch import no_grad, cuda, save
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from src.algorithm.ml_model.MLModel import MLModel
from src.algorithm.ml_model.models.Autoencoder import Autoencoder
from src.config.ConfigParams import ConfigParams
from src.algorithm.ml_model.History import History


class TorchModel(MLModel):

    def __init__(self, identifier: str, input_length: int, learning_rate: float):
        super().__init__(identifier)

        self.device = TorchModel.__get_device()

        self.model = Autoencoder(input_length)
        self.model.to(self.device)

        self.criterion = MSELoss()
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
    def load(config: ConfigParams, path: str):
        pass

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
            loss = self.__run_epoch(trainloader)
            train_error.append(loss.item())

            if validationloader is not None:
                validation_loss = self.__run_epoch(validationloader, is_train=False)
                validation_error.append(validation_loss.item())

            if (epoch % 5) == 0:
                print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss.item(): .4f}')

            learning_rate_updating.append(config.get_params_dict('train_params')['learning_rate'])

        return History(train_error, validation_error, learning_rate_updating)

    def __run_epoch(self, dataloader: DataLoader, is_train=True):
        loss = None

        for trainbatch in dataloader:
            batch = trainbatch.to(self.device)

            output = self.model(batch)
            loss = self.criterion(output, batch.data)

            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss

    # TODO: Check this function
    def test(self, config: dict):
        testloader = DataLoader([])
        test_error = []

        with no_grad():
            test_error = self.__run_epoch(testloader, False)
