from torch.nn import Module, Sequential, Linear, Tanh

from src.algorithm.ml_model.MLModel import MLModel


class Autoencoder(MLModel, Module):

    def __init__(self, input_length):
        super(MLModel).__init__("")
        super(Module).__init__()
        self.encoder = Sequential(
            Linear(input_length, 256),
            Tanh(),
            Linear(256, 128),
            Tanh(),
        )

        self.decoder = Sequential(
            Linear(128, 256),
            Tanh(),
            Linear(256, input_length),
            Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    @staticmethod
    def create(config: dict) -> MLModel:
        input_length = 1000
        ae_instance = Autoencoder(input_length)
        return ae_instance

    @staticmethod
    def load(config: dict):
        pass

    def save(self, config: dict):
        pass

    def train(self, config: dict):
        pass

    def test(self, config: dict):
        pass
