import os

import numpy as np

from src.config.CommonPath import CommonPath


def stack_arrays(stacked_array: np.ndarray, new_array: np.ndarray) -> np.ndarray:
    if stacked_array.shape[0] != 0:
        return np.vstack((stacked_array, new_array))
    else:
        return new_array


def build_model_folderpath(model_identifier: str, config_identifier: str):
    model_folder = f'{model_identifier}_cnf_{config_identifier}'
    return os.path.join(CommonPath.MODEL_PARAMETERS_FOLDER.value, model_folder)
