import numpy as np
from torch import from_numpy, float32, dtype, Tensor
from torch.utils.data import Dataset


class CustomTorchDataset(Dataset):
    """
    A custom class that wraps a NumPy array (implements the abstract class Dataset from Pytorch).
    """
    def __init__(self, data: np.ndarray, data_type: dtype = float32):
        """
        Initializes an instance of CustomTorchDataset.
        @param data The dataset as a NumPy array.
        @param data_type The data type of the PyTorch tensor. Default is float32.
        """
        self.data: Tensor = from_numpy(data).to(data_type)

    def __getitem__(self, index):
        """
        Returns an item from the dataset.
        @param index The index of the item to retrieve.
        """
        return self.data[index]

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.data)
