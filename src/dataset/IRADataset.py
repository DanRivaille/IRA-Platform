from abc import ABC

import numpy as np
from torch.utils.data import Dataset

from src.dataset.dataset_type import DatasetType
from src.dataset.CustomTorchDataset import CustomTorchDataset


class IRADataset(ABC):
    def __init__(self, data: np.ndarray, type_dataset: DatasetType):
        self.data: np.ndarray = data
        self.type_dataset: DatasetType = type_dataset

    def get_torch_dataset(self) -> Dataset:
        return CustomTorchDataset(self.data)
