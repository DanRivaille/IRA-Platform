from abc import ABC, abstractmethod

from src.dataset.IRADataset import IRADataset


class PreprocessStep(ABC):
    """
    An abstract class representing a preprocess step for the IRADataset.
    """
    @abstractmethod
    def apply(self, dataset: IRADataset):
        """
        Method to apply the preprocess step to a given IRADataset (Abstract method).
        @param dataset A class for loading the datasets used by models.
        """
        pass

    @abstractmethod
    def save(self, folder: str):
        """
        Saves the preprocess step to a specified folder.
        @param folder The folder path where the preprocess step should be saved.
        """
        pass

    @abstractmethod
    def load(self, folder: str):
        """
        Loads the preprocess step from a specified folder.
        @param folder The folder path from which the preprocess step should be loaded.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_filename() -> str:
        """
        Returns the filename associated with the preprocess step.
        """
        pass
