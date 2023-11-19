from torch.utils.data import Dataset
from torch import from_numpy, float32
import numpy as np


class CustomTorchDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = from_numpy(data).to(float32)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
