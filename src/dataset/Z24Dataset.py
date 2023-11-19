import os
from datetime import datetime, timedelta

import numpy as np
from scipy.io import loadmat

from src.Utils.utils import stack_arrays

DATA_FOLDER = '/work/ivan.santos/datasets/z24/mat_files'


class Z24Dataset:

    def __init__(self, data: np.ndarray):
        self.data = data

    @staticmethod
    def load(first_date: datetime, last_date: datetime, sensor_number: int):
        total_hours = (last_date - first_date + timedelta(days=1)) // timedelta(hours=1)
        year = str(first_date.year)
        data = np.empty(0)

        for current_hour in range(total_hours):
            current_datetime = first_date + timedelta(hours=current_hour)

            filename = f'd_{year[2:]}_{current_datetime.month}_{current_datetime.day}_{current_datetime.hour}.mat'
            file_path = os.path.join(DATA_FOLDER, filename)

            if os.path.isfile(file_path):
                data_mat = loadmat(file_path)
                new_data = data_mat['Data'][sensor_number]

                data = stack_arrays(data, new_data)

        return Z24Dataset(data)
