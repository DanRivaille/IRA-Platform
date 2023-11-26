from keras import Model

from src.config.ConfigParams import ConfigParams


class KerasLoader:
    def __init__(self, config_params: ConfigParams):
        self.__config = config_params

    def load(self) -> Model:
        pass
