from keras import Model, Sequential
from keras.src.layers import Dense, LSTM

from src.config.ConfigParams import ConfigParams


class KerasLoader:
    def __init__(self, config_params: ConfigParams):
        self.__config: config_params = config_params
        self.__topology: [dict] = self.__config.get_params_dict('network_params').get('topology')

    def load(self) -> Model:
        model = Sequential()

        for layer in self.__topology:
            model.add(KerasLoader.__load_layer(layer))

        return model

    @staticmethod
    def __load_layer(layer_info: dict):
        pass

    @staticmethod
    def __get_layer(layer_code: str):
        if "dense_keras_layer" == layer_code:
            return Dense
        elif "lstm_keras_layer" == layer_code:
            return LSTM
