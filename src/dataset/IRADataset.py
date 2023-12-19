from abc import ABC

import numpy as np
from torch.utils.data import DataLoader

from src.algorithm.ml_model.ModelType import ModelType
from src.config.ConfigParams import ConfigParams
from src.dataset.CustomTorchDataset import CustomTorchDataset
from src.dataset.dataset_type import DatasetType


class IRADataset(ABC):
    def __init__(self, data: np.ndarray, type_dataset: DatasetType):
        self.data: np.ndarray = data
        self.type_dataset: DatasetType = type_dataset

    def get_dataloader(self, model_type: ModelType, batch_size: int) -> DataLoader | np.ndarray:
        if ModelType.KERAS_MODEL is model_type:
            return self.__get_keras_dataloader()
        else:
            return self.__get_torch_dataloader(batch_size)

    def __get_torch_dataloader(self, batch_size: int) -> DataLoader:
        return DataLoader(CustomTorchDataset(self.data), batch_size=batch_size)

    def __get_keras_dataloader(self) -> np.ndarray:
        return self.data

    @staticmethod
    def load(config: ConfigParams, type_dataset: DatasetType):
        pass
