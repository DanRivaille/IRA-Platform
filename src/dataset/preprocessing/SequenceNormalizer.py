import os.path

from sklearn.preprocessing import MinMaxScaler
import joblib

from src.dataset.IRADataset import IRADataset
from src.dataset.dataset_type import DatasetType
from src.dataset.preprocessing.PreprocessStep import PreprocessStep


class SequenceNormalizer(PreprocessStep):
    __scaler: MinMaxScaler | None = None

    def __init__(self, new_range: tuple):
        self.__new_range: tuple = new_range
        SequenceNormalizer.__scaler = MinMaxScaler(feature_range=self.__new_range)

    @staticmethod
    def get_filename() -> str:
        pass

    def load(self, folder: str):
        pass

    def save(self, folder: str):
        pass

    def apply(self, dataset: IRADataset):
        data = dataset.data
        dataset.data = SequenceNormalizer.__scaler.fit_transform(data.T).T
