import numpy as np
from torch import from_numpy, float32, dtype, Tensor
from torch.utils.data import Dataset


class CustomTorchDataset(Dataset):
    def __init__(self, data: np.ndarray, data_type: dtype = float32):
        self.data: Tensor = from_numpy(data).to(data_type)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
