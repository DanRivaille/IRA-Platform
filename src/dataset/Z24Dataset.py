import os
from datetime import datetime, timedelta

import numpy as np
from scipy.io import loadmat

from src.dataset.IRADataset import IRADataset
from src.utils.utils import stack_arrays
from src.config.ConfigParams import ConfigParams
from src.config.CommonPath import CommonPath
from src.dataset.dataset_type import DatasetType


class Z24Dataset(IRADataset):
    def __init__(self, data: np.ndarray, type_dataset: DatasetType):
        super().__init__(data, type_dataset)

    @staticmethod
    def load(config: ConfigParams, type_dataset: DatasetType):
        data_params = config.get_params_dict(type_dataset.value)
        first_date = datetime.strptime(data_params.get('first_date'), "%d/%m/%Y")
        last_date = datetime.strptime(data_params.get('last_date'), "%d/%m/%Y")
        sensor_number = config.get_params_dict('preprocess_params').get('sensor_number')

        total_hours = (last_date - first_date + timedelta(days=1)) // timedelta(hours=1)
        year = str(first_date.year)
        data = np.empty(0)

        for current_hour in range(total_hours):
            current_datetime = first_date + timedelta(hours=current_hour)

            foldername = f'{str(current_datetime.month).zfill(2)}{str(current_datetime.day).zfill(2)}'
            filename = f'd_{year[2:]}_{current_datetime.month}_{current_datetime.day}_{current_datetime.hour}.mat'
            file_path = os.path.join(CommonPath.DATA_FOLDER.value, foldername, filename)

            if os.path.isfile(file_path):
                data_mat = loadmat(file_path)
                new_data = data_mat['Data'][:, sensor_number]

                data = stack_arrays(data, new_data)

        return Z24Dataset(data, type_dataset)
