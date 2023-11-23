from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import DataLoader

from src.algorithm.Algorithm import Algorithm
from src.algorithm.ml_model.History import History
from src.config.ConfigParams import ConfigParams


class MLModel(Algorithm, ABC):

    def __init__(self, identifier: str):
        super().__init__(identifier)

    """
    Get the file extension for the file with the parameters of the model
    """
    @staticmethod
    @abstractmethod
    def get_file_extension():
        pass

    """
    Runs the train process
    """
    @abstractmethod
    def train(self, config: ConfigParams, trainloader: DataLoader, validationloader: DataLoader | None) -> History:
        pass

    """
    Runs the test process
    """
    @abstractmethod
    def test(self, config: ConfigParams, testloader: DataLoader, validationloader: DataLoader):
        pass

    """
    Predict the data in dataloader using the current state of the model
    @return Tuple: (predictions, errors per sample)
    """
    @abstractmethod
    def predict(self, dataloader: DataLoader, **kwargs) -> tuple:
        pass
