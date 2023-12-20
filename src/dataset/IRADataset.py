from abc import ABC

import numpy as np
from torch.utils.data import DataLoader

from src.algorithm.ml_model.ModelType import ModelType
from src.config.ConfigParams import ConfigParams
from src.dataset.CustomTorchDataset import CustomTorchDataset
from src.dataset.dataset_type import DatasetType


class IRADataset(ABC):
    """
    An abstract class for loading datasets used by models.
    """
    def __init__(self, data: np.ndarray, type_dataset: DatasetType):
        """
        Initializes an instance of IRADataset.
        @param data An array stack with the data from a dataset.
        @param type_dataset The type of dataset that the model will use (TRAIN, TEST, VALIDATION TRAIN o VALIDATION TEST).
        """
        self.data: np.ndarray = data
        self.type_dataset: DatasetType = type_dataset

    def get_dataloader(self, model_type: ModelType, batch_size: int) -> DataLoader | np.ndarray:
        """
        Returns the supported data type for the model. A DataLoader for PyTorch models or a NumPy array for Keras models.
        @param model_type The type of the machine learning model used.
        @param batch_size The batch size for the DataLoader.
        """
        if ModelType.KERAS_MODEL is model_type:
            return self.__get_keras_dataloader()
        else:
            return self.__get_torch_dataloader(batch_size)

    def __get_torch_dataloader(self, batch_size: int) -> DataLoader:
        """
        Returns a DataLoader for PyTorch models.
        @param batch_size The batch size for the DataLoader.
        """
        return DataLoader(CustomTorchDataset(self.data), batch_size=batch_size)

    def __get_keras_dataloader(self) -> np.ndarray:
        """
        Returns the NumPy array for Keras models.
        """
        return self.data

    @staticmethod
    def load(config: ConfigParams, type_dataset: DatasetType):
        """
        Loads a dataset based on configuration parameters and dataset type (Abstract method).
        @param config Configuration parameters for the dataset loaded from a .json file (they are the first date and last date).
        @param type_dataset The type of dataset that the model will use (TRAIN, TEST, VALIDATION TRAIN o VALIDATION TEST).
        """
        pass
