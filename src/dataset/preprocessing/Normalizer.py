import os.path

import joblib
from sklearn.preprocessing import MinMaxScaler

from src.dataset.IRADataset import IRADataset
from src.dataset.dataset_type import DatasetType
from src.dataset.preprocessing.PreprocessStep import PreprocessStep


class Normalizer(PreprocessStep):
    """
    A preprocess step that applies Min-Max normalization to the data in an IRADataset.
    """

    __scaler: MinMaxScaler | None = None

    def __init__(self, new_range: tuple):
        """
        Initializes an instance of Normalizer.
        @param new_range The desired range for the normalized values.
        """
        self.__new_range = new_range

    def apply(self, dataset: IRADataset):
        """
        Applies Min-Max normalization to the data in the given IRADataset.
        @param dataset A class for loading the datasets used by models.
        """
        data = dataset.data
        original_shape = data.shape
        data_to_transform = data.reshape((-1, 1))

        if dataset.type_dataset == DatasetType.TRAIN_DATA:
            Normalizer.__scaler = MinMaxScaler(feature_range=self.__new_range)
            Normalizer.__scaler.fit(data_to_transform)

        dataset.data = Normalizer.__scaler.transform(data_to_transform).reshape(original_shape)

    def save(self, folder: str):
        """
        Saves the preprocess step to a specified folder.
        @param folder The folder path where the preprocess step should be saved.
        """
        file_path = os.path.join(folder, Normalizer.get_filename())
        joblib.dump(Normalizer.__scaler, file_path)

    def load(self, folder: str):
        """
        Loads the preprocess step from a specified folder.
        @param folder The folder path from which the preprocess step should be loaded.
        """
        file_path = os.path.join(folder, Normalizer.get_filename())
        if os.path.isfile(file_path):
            Normalizer.__scaler = joblib.load(file_path)

    @staticmethod
    def get_filename() -> str:
        """
        Returns the filename associated with the preprocess step.
        """
        return 'normalizer.gz'
