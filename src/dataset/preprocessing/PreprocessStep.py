from abc import ABC, abstractmethod

from src.dataset.IRADataset import IRADataset


class PreprocessStep(ABC):
    @abstractmethod
    def apply(self, dataset: IRADataset):
        pass

    @abstractmethod
    def save(self, folder: str):
        pass

    @abstractmethod
    def load(self, folder: str):
        pass

    @staticmethod
    @abstractmethod
    def get_filename() -> str:
        pass
