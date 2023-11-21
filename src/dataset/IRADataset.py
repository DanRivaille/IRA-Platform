from abc import ABC

import numpy as np

from src.dataset.dataset_type import DatasetType


class IRADataset(ABC):
    def __init__(self, data: np.ndarray, type_dataset: DatasetType):
        self.data: np.ndarray = data
        self.type_dataset: DatasetType = type_dataset
