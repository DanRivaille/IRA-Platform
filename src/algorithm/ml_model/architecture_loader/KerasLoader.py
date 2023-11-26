from keras import Model, Sequential
from keras.src.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.optimizers import Adam

from src.config.ConfigParams import ConfigParams


class KerasLoader:
    def __init__(self, config_params: ConfigParams):
        self.config_params = config_params

    def load(self) -> Model:
        topology = self.config_params.get_params_dict('network_params').get('topology')

        model = Sequential()

        for layer in topology:
            model.add(KerasLoader.__load_layer(layer))

        compilation_params = KerasLoader.__load_compilation_params(self.config_params)
        #model.compile(**compilation_params)
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='mean_squared_error')

        return model

    @staticmethod
    def __load_compilation_params(config_params: ConfigParams) -> dict:
        network_params = config_params.get_params_dict('network_params')
        optimizer = KerasLoader.__get_optimizer(network_params.get('optimizer'))
        learning_rate = config_params.get_params_dict('train_params')['learning_rate']

        print(network_params.get('loss_function'))
        return {
            'optimizer': optimizer(learning_rate=learning_rate),
            'loss': network_params.get('loss_function')
        }

    @staticmethod
    def __get_optimizer(optimizer_code: str):
        if "adam" == optimizer_code:
            return Adam

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
        KerasLoader.__add_layer_param(layer_info, layer_params, 'activation')
        KerasLoader.__add_layer_param(layer_info, layer_params, 'return_sequences')

        return layer_params

    @staticmethod
    def __add_layer_param(layer_info: dict, layer_params: dict, keyparam_to_add: str) -> None:
        param_to_add = layer_info.get(keyparam_to_add, None)
        if param_to_add is not None:
            layer_params[keyparam_to_add] = param_to_add
