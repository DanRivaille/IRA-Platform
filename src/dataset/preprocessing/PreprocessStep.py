from abc import ABC, abstractmethod

from src.dataset.IRADataset import IRADataset


class PreprocessStep(ABC):
    @abstractmethod
    def apply(self, dataset: IRADataset):
        pass

