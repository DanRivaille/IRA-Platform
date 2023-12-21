from json import dump

import numpy as np


class History:
    """
    A class for saving the training history data of model execution.
    """
    def __init__(self, train_loss: list, valid_loss: list, learning_rate: list, train_error_per_sample: np.ndarray):
        """
        Initializes an instance of History.
        @param train_loss List of training losses for the model.
        @param valid_loss List of validation losses for the model.
        @param learning_rate List of learning rates during the training of the model.
        @param train_error_per_sample An array containing the errors associated with each individual sample in the training set.
        """
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.learning_rate = learning_rate
        self.train_error_per_sample: np.ndarray = train_error_per_sample

    def to_json(self) -> dict:
        """
        Returns a dictionary for a .json file with the training history. 
        """
        entries = {
            'train_loss': self.train_loss,
            'valid_loss': self.valid_loss,
            'learning_rate_updating': self.learning_rate
        }

        return entries

    def save(self, path: str):
        """
        Saves the training history to a .json file.
        @param path The file path where the results will be saved.
        """
        with open(path, 'w') as history_file:
            dump(self.to_json(), history_file)

        np.savez_compressed(path, error=self.train_error_per_sample)
