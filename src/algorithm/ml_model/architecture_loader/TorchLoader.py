from torch.nn import Module, Sequential

from src.config.ConfigParams import ConfigParams


class TorchLoader:
    def __init__(self, config_params: ConfigParams):
        self.__config: config_params = config_params
        self.__topology: [dict] = self.__config.get_params_dict('network_params').get('topology')

    def load(self) -> Module:
        model = Sequential()

        for layer in self.__topology:
            pass

        return model
