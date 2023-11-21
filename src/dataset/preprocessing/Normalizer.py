import os.path

from sklearn.preprocessing import MinMaxScaler
import joblib

from src.dataset.IRADataset import IRADataset
from src.dataset.dataset_type import DatasetType
from src.dataset.preprocessing.PreprocessStep import PreprocessStep


class Normalizer(PreprocessStep):
    __scaler: MinMaxScaler | None = None

    def __init__(self, new_range: tuple):
        self.__new_range = new_range

    def apply(self, dataset: IRADataset):
        data = dataset.data
        original_shape = data.shape
        data_to_transform = data.reshape((-1, 1))

        if dataset.type_dataset == DatasetType.TRAIN_DATA:
            Normalizer.__scaler = MinMaxScaler(feature_range=self.__new_range)
            Normalizer.__scaler.fit(data_to_transform)

        normalized_data = Normalizer.__scaler.transform(data_to_transform).reshape(original_shape)
        dataset.data = normalized_data

    def save(self, folder: str):
        file_path = os.path.join(folder, Normalizer.get_filename())
        joblib.dump(Normalizer.__scaler, file_path)

    def load(self, folder: str):
        file_path = os.path.join(folder, Normalizer.get_filename())
        Normalizer.__scaler = joblib.load(file_path)

    @staticmethod
    def get_filename() -> str:
        return 'normalizer.gz'
