import numpy as np


def stack_arrays(stacked_array: np.ndarray, new_array: np.ndarray) -> np.ndarray:
    if stacked_array.shape[0] != 0:
        return np.vstack((stacked_array, new_array))
    else:
        return new_array
