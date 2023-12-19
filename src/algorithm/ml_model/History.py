from json import dump

import numpy as np


class History:
    def __init__(self, train_loss: list, valid_loss: list, learning_rate: list, train_error_per_sample: np.ndarray):
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.learning_rate = learning_rate
        self.train_error_per_sample: np.ndarray = train_error_per_sample

    def to_json(self) -> dict:
        entries = {
            'train_loss': self.train_loss,
            'valid_loss': self.valid_loss,
            'learning_rate_updating': self.learning_rate
        }

        return entries

    def save(self, path: str):
        with open(path, 'w') as history_file:
            dump(self.to_json(), history_file)

        np.savez_compressed(path, error=self.train_error_per_sample)
