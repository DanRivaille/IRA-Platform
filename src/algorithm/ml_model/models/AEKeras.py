from keras import Sequential, Model
from keras.layers import Dense


def get_autoencoder_keras(input_size: int, optimizer, loss_function) -> Model:
    """
    Constructs a autoencoder model using Keras.
    @param input_size The size of each input and output layer entry.
    @param optimizer The optimizer to be used during model compilation.
    @param loss_function The loss function to be used during model compilation.
    """
    encoding_dim = 128
    model = Sequential()

    # Encoder
    model.add(Dense(256, input_shape=(input_size,)))
    model.add(Dense(encoding_dim))

    # Decoder
    model.add(Dense(256))

    # Output layer
    model.add(Dense(input_size))

    model.compile(optimizer=optimizer, loss=loss_function)
    return model
