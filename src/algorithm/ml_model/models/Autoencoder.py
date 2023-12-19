from torch.nn import Module, Sequential, Linear, Tanh


class Autoencoder(Module):
    def __init__(self, input_length):
        super().__init__()
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
