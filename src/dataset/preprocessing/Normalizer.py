from sklearn.preprocessing import MinMaxScaler

from src.dataset.dataset_type import DatasetType
from src.dataset.preprocessing.PreprocessStep import PreprocessStep


class Normalizer(PreprocessStep):
    __scaler: MinMaxScaler | None = None

    def __init__(self, new_range: tuple):
        self.__new_range = new_range

    def apply(self, data, data_type: DatasetType | None = None):
        original_shape = data.shape
        data_to_transform = data.reshape((-1, 1))

        if data_type == DatasetType.TRAIN_DATA:
            Normalizer.__scaler = MinMaxScaler(feature_range=self.__new_range)
            Normalizer.__scaler.fit(data_to_transform)

        return Normalizer.__scaler.transform(data_to_transform).reshape(original_shape)