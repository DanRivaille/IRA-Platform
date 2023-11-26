from keras import Model, Sequential
from keras.src.layers import Dense, LSTM, RepeatVector, TimeDistributed

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
        layer = KerasLoader.__get_layer(layer_info.get('layer'))

        if layer == 'time_distributed_keras_layer':
            return layer(KerasLoader.__load_layer(layer_info['time_distributed_layer']))

        layer_params = KerasLoader.__get_layer_params(layer_info)
        n_neurons = layer_info.get('neurons')

        return layer(n_neurons, **layer_params)

    @staticmethod
    def __get_layer(layer_code: str):
        if "dense_keras_layer" == layer_code:
            return Dense
        elif "lstm_keras_layer" == layer_code:
            return LSTM
        elif "repeat_vector_keras_layer" == layer_code:
            return RepeatVector
        elif "time_distributed_keras_layer" == layer_code:
            return TimeDistributed

    @staticmethod
    def __get_layer_params(layer_info: dict) -> dict:
        layer_params = {}

        KerasLoader.__add_layer_param(layer_info, layer_params, 'input_shape')
        KerasLoader.__add_layer_param(layer_info, layer_params, 'activation_function')
        KerasLoader.__add_layer_param(layer_info, layer_params, 'return_sequences')

        return layer_params

    @staticmethod
    def __add_layer_param(layer_info: dict, layer_params: dict, keyparam_to_add: str) -> None:
        param_to_add = layer_info.get(keyparam_to_add, None)
        if param_to_add is not None:
            layer_params[keyparam_to_add] = param_to_add
