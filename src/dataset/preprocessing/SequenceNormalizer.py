from sklearn.preprocessing import MinMaxScaler

from src.dataset.IRADataset import IRADataset
from src.dataset.preprocessing.PreprocessStep import PreprocessStep


class SequenceNormalizer(PreprocessStep):
    """
    A preprocess step that applies Min-Max normalization to the sequences in the dataset.
    """
    __scaler: MinMaxScaler | None = None

    def __init__(self, new_range: tuple):
        """
        Initializes an instance of SequenceNormalizer.
        @param new_range The desired range for the normalized values.
        """
        self.__new_range: tuple = new_range
        SequenceNormalizer.__scaler = MinMaxScaler(feature_range=self.__new_range)

    @staticmethod
    def get_filename() -> str:
        """
        Returns the filename associated with the preprocess step.
        """
        pass

    def load(self, folder: str):
        """
        Loads the preprocess step from a specified folder.
        @param folder The folder path from which the preprocess step should be loaded.
        """
        pass

    def save(self, folder: str):
        """
        Saves the preprocess step to a specified folder.
        @param folder The folder path where the preprocess step should be saved.
        """
        pass

    def apply(self, dataset: IRADataset):
        """
        Applies Min-Max normalization to the sequences in the dataset.
        @param dataset A class for loading the datasets used by models.
        """
        data = dataset.data
        dataset.data = SequenceNormalizer.__scaler.fit_transform(data.T).T
