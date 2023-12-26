from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import DataLoader

from src.algorithm.Algorithm import Algorithm
from src.algorithm.ml_model.History import History
from src.algorithm.ml_model.ModelType import ModelType
from src.config.ConfigParams import ConfigParams


class MLModel(Algorithm, ABC):
    """
    An abstract class representing a machine learning model. 
    """
    def __init__(self, identifier: str):
        """
        Initializes an instance of MLModel.
        @param identifier An identifier for the model.
        """
        super().__init__(identifier)

    
    @staticmethod
    @abstractmethod
    def get_file_extension():
        """
        Get the file extension for the file with the parameters of the model (Abstract method).
        """
        pass

    @staticmethod
    @abstractmethod
    def get_model_type() -> ModelType:
        """
        Get the model type (Abstract method).
        """
        pass

    
    @abstractmethod
    def train(self,
              config: ConfigParams,
              trainloader: DataLoader | np.ndarray,
              validationloader: DataLoader | np.ndarray | None) -> History:
        """
        Runs the train process (Abstract method).
        @param config A class for loading and saving configuration parameters related to a model.
        @param trainloader DataLoader or numpy array containing training data.
        @param validationloader DataLoader or numpy array containing validation data.
        @return History A class for saving the training history data of model execution.
        """
        pass

    
    @abstractmethod
    def test(self,
             config: ConfigParams,
             testloader: DataLoader | np.ndarray,
             validationloader: DataLoader | np.ndarray):
        """
        Runs the test process (Abstract method).
        @param config A class for loading and saving configuration parameters related to a model.
        @param testloader DataLoader or numpy array containing test data.
        @param validationloader DataLoader or numpy array containing validation data.
        """
        pass

    
    @abstractmethod
    def predict(self, dataloader: DataLoader | np.ndarray, **kwargs) -> tuple:
        """
        Predict the data in dataloader using the current state of the model (Abstract method).
        @param dataloader DataLoader or NumPy array containing the data to predict.
        @return Tuple: (predictions, errors per sample)
        """
        pass
