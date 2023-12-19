import os

import numpy as np

from src.config.CommonPath import CommonPath

def stack_arrays(stacked_array: np.ndarray, new_array: np.ndarray) -> np.ndarray:
    """
    Stacks an array inside an array stack, if the array stack is empty then returns the new array.
    @param stacked_array Current array stack (may be empty).
    @param new_array A new array to enter the stack.
    """
    if stacked_array.shape[0] != 0:
        return np.vstack((stacked_array, new_array))
    else:
        return new_array


def build_model_folderpath(model_identifier: str, config_identifier: str):
    """
    Builds a directory path for a model and its configuration based on the provided identifiers
    @param model_identifier A string identifier for the current model.
    @param config_identifier A string identifier of the current model configuration. 
    """
    model_folder = f'{model_identifier}_cnf_{config_identifier}'
    return os.path.join(CommonPath.MODEL_PARAMETERS_FOLDER.value, model_folder)
