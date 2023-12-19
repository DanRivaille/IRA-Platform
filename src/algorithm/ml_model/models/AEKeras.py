from keras import Sequential, Model
from keras.layers import Dense


def get_autoencoder_keras(input_size: int, optimizer, loss_function) -> Model:
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
