from keras import Model, Sequential
from keras.src.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.optimizers import Adam

from src.config.ConfigParams import ConfigParams


class KerasLoader:
    """
    A class responsible for loading a Keras model based on configuration parameters.
    """
    def __init__(self, config_params: ConfigParams):
        """
        Initializes an instance of KerasLoader.
        @param config_params A class for loading configuration parameters related to a model.
        """
        self.config_params = config_params

    def load(self) -> Model:
        """
        Loads a Keras model with the layers specified in the provided configuration parameters.
        """
        topology = self.config_params.get_params_dict('network_params').get('topology')

        model = Sequential()

        for layer in topology:
            model.add(KerasLoader.__load_layer(layer))

        compilation_params = KerasLoader.__load_compilation_params(self.config_params)
        model.compile(**compilation_params)

        return model

    @staticmethod
    def __load_compilation_params(config_params: ConfigParams) -> dict:
        """
        Loads a Keras model with compilation params specified in the provided configuration.
        @param config_params A class for loading configuration parameters related to a model.
        @return Dict: With the compilation params (optimizer and loss function).
        """
        network_params = config_params.get_params_dict('network_params')
        optimizer = KerasLoader.__get_optimizer(network_params.get('optimizer'))
        learning_rate = config_params.get_params_dict('train_params')['learning_rate']

        return {
            'optimizer': optimizer(learning_rate=learning_rate),
            'loss': network_params.get('loss_function')
        }

    @staticmethod
    def __get_optimizer(optimizer_code: str):
        """
        Returns the optimizer class based on the provided optimizer code.
        @param optimizer_code The code of the optimizer to return.
        """
        if "adam" == optimizer_code:
            return Adam

    @staticmethod
    def __load_layer(layer_info: dict):
        """
        Loads and returns a Keras layer based on the provided layer information.
        @param layer_info A dictionary of the layers to load.
        """
        layer_name = layer_info.get('layer')
        layer = KerasLoader.__get_layer(layer_name)

        if layer_name == 'time_distributed_keras_layer':
            return layer(KerasLoader.__load_layer(layer_info['time_distributed_layer']))

        layer_params = KerasLoader.__get_layer_params(layer_info)
        n_neurons = layer_info.get('neurons')

        return layer(n_neurons, **layer_params)

    @staticmethod
    def __get_layer(layer_code: str):
        """
        Returns the Keras layer class based on the provided layer code.
        @param layer_code The code of the layer to return.
        """
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
        """
        Gets and returns the layer parameters based on the provided layer information.
        @param layer_info Dictionary with information about the layer.
        """
        layer_params = {}

        KerasLoader.__add_layer_param(layer_info, layer_params, 'input_shape')
        KerasLoader.__add_layer_param(layer_info, layer_params, 'activation')
        KerasLoader.__add_layer_param(layer_info, layer_params, 'return_sequences')

        return layer_params

    @staticmethod
    def __add_layer_param(layer_info: dict, layer_params: dict, keyparam_to_add: str) -> None:
        """
        Adds a layer parameter to the layer_params dictionary if it exists in layer_info.
        @param layer_info Dictionary with information about the layer.
        @param layer_params Dictionary to store layer parameters.
        @param keyparam_to_add Key parameter to add to layer_params.
        """
        param_to_add = layer_info.get(keyparam_to_add, None)
        if param_to_add is not None:
            layer_params[keyparam_to_add] = param_to_add
