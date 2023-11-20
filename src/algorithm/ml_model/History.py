import os
from json import dump


class History:
    def __init__(self, train_loss: list, valid_loss: list, learning_rate: list):
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.learning_rate = learning_rate

    def to_json(self) -> dict:
        entries = {
            'train_loss': self.train_loss,
            'valid_loss': self.valid_loss,
            'learning_rate_updating': self.learning_rate
        }

        return entries

    def save(self, folder: str):
        with open(os.path.join(folder, 'history.json'), 'w') as history_file:
            dump(self.to_json(), history_file)
