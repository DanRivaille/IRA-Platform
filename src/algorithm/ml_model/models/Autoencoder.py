from torch.nn import Module, Sequential, Linear, Tanh


class Autoencoder(Module):
    """
    A class of an autoencoder model using PyTorch.
    """
    def __init__(self, input_length):
        """
        Initializes an instance of Autoencoder.
        @param input_length The size of each input and output layer entry.
        """
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
        """
        Performs the forward pass of the autoencoder.
        @param x The input tensor to be encoded and then decoded.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x
