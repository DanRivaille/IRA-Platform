import os
from datetime import datetime, timedelta

from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

from src.dataset.CustomTorchDataset import CustomTorchDataset
from src.utils.utils import stack_arrays

DATA_FOLDER = '/work/ivan.santos/datasets/z24/mat_files'


class Z24Dataset:
    scaler = MinMaxScaler(feature_range=(-1, 1))

    def __init__(self, data: np.ndarray):
        self.data = data

    @staticmethod
    def load(first_date: datetime, last_date: datetime, sensor_number: int):
        total_hours = (last_date - first_date + timedelta(days=1)) // timedelta(hours=1)
        year = str(first_date.year)
        data = np.empty(0)

        for current_hour in range(total_hours):
            current_datetime = first_date + timedelta(hours=current_hour)

            foldername = f'{str(current_datetime.month).zfill(2)}{str(current_datetime.day).zfill(2)}'
            filename = f'd_{year[2:]}_{current_datetime.month}_{current_datetime.day}_{current_datetime.hour}.mat'
            file_path = os.path.join(DATA_FOLDER, foldername, filename)

            if os.path.isfile(file_path):
                data_mat = loadmat(file_path)
                new_data = data_mat['Data'][:, sensor_number]

                data = stack_arrays(data, new_data)

        return Z24Dataset(data)

    def normalize_data(self, is_train_data: bool = True, inplace: bool = False):
        original_shape = self.data.shape
        data_to_transform = self.data.reshape((-1, 1))

        if is_train_data:
            Z24Dataset.scaler.fit(data_to_transform)

        data_normalized = Z24Dataset.scaler.transform(data_to_transform).reshape(original_shape)
        if inplace:
            self.data = data_normalized
        else:
            return Z24Dataset(data_normalized)

    def reshape_in_sequences(self, sequences_length: int, inplace: bool = False):
        n_samples, sample_length = self.data.shape

        sequence_samples_to_consider = (sample_length // sequences_length) * sequences_length
        new_data = self.data[:, :sequence_samples_to_consider].reshape((-1, sequences_length))

        if inplace:
            self.data = new_data
        else:
            return Z24Dataset(new_data)

    def get_torch_dataset(self) -> Dataset:
        return CustomTorchDataset(self.data)
