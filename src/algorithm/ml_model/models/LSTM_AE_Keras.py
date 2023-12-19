from keras import Model, Sequential
from keras.src.layers import LSTM, RepeatVector, TimeDistributed, Dense


def get_lstm_ae_keras(input_size: int, n_features: int, optimizer, loss_function: str, metric_list=None) -> Model:
    if None is metric_list:
        metric_list = []

    model = Sequential()

    # Encoder
    model.add(LSTM(128, input_shape=(input_size, n_features), return_sequences=False))

    # Encoded layer (bottleneck)
    model.add(RepeatVector(input_size))

    # Decoder
    model.add(LSTM(128, input_shape=(input_size, n_features), return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metric_list)

    return model


def get_lstm_ae_keras_modify(optimizer, loss_function: str) -> Model:
    model = Sequential()

    model.add(LSTM(128, input_shape=(50, 1), return_sequences=False))
    model.add(RepeatVector(50))
    model.add(LSTM(128, input_shape=(50, 1), return_sequences=True))
    model.add(TimeDistributed(Dense(1)))

    model.compile(optimizer=optimizer, loss=loss_function)

    return model
