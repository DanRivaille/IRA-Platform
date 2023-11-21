from abc import ABC, abstractmethod

from src.dataset.dataset_type import DatasetType


class PreprocessStep(ABC):
    @abstractmethod
    def apply(self, data, data_type: DatasetType | None = None):
        pass

