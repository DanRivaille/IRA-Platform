from keras import Sequential, Model
from keras.layers import Dense


def get_autoencoder_keras(input_size: int, optimizer, loss_function, metric_list=None) -> Model:
    if metric_list is None:
        metric_list = []

    encoding_dim = 128
    model = Sequential()

    # Encoder
    model.add(Dense(256, activation='tanh', input_shape=(input_size,)))
    model.add(Dense(encoding_dim, activation='tanh'))

    # Decoder
    model.add(Dense(256, activation='tanh'))

    # Output layer
    model.add(Dense(input_size, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metric_list)
    return model


def get_autoencoder_keras_modify(optimizer, loss_function, metric_list=None) -> Model:
    model = Sequential()

    # Encoder
    model.add(Dense(256, activation='tanh', input_shape=(1000,)))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(1000, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metric_list)
    return model

