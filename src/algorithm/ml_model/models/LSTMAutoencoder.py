from torch.nn import Module, LSTM


class LSTMAutoencoder(Module):
    """
    A class of an LSTM-Autoencoder model using PyTorch.
    """
    def __init__(self, input_size = 1, hidden_size = 128, num_layers = 1):
        """
        Initializes an instance of LSTMAutoencoder.
        @param input_size The number of expected features in the input.
        @param hidden_size The number of features in the hidden state of the LSTM.
        @param num_layers Number of recurrent layers.
        """
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        """
        Performs the forward pass of the LSTMAutoencoder.
        @param x The input tensor to be encoded and then decoded.
        """
        x, _ = self.encoder(x)
        x, _ = self.decoder(x)
        return x