from abc import ABC

from torch import device, no_grad
from torch.optim import Optimizer
from torch.nn import Module, MSELoss
from torch.utils.data import DataLoader

from src.algorithm.ml_model.MLModel import MLModel


class TorchModel(MLModel, ABC):

    def __init__(self):
        super(MLModel).__init__("")
        self.model = Module()
        self.device = device
        self.criterion = MSELoss
        self.optimizer = Optimizer()

    @staticmethod
    def load(config: dict):
        pass

    def save(self, config: dict):
        pass

    def train(self, config: dict):
        num_epochs = config['train_params']['num_epochs']
        trainloader = DataLoader([])
        validationloader = DataLoader([])

        train_error = []
        validation_error = []

        for epoch in range(num_epochs):
            loss = self.__run_epoch(trainloader)
            validation_loss = self.__run_epoch(validationloader, False)

            train_error.append(loss.item())
            validation_error.append(validation_loss.item())

            if (epoch % 5) == 0:
                print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss.item(): .4f}')

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
