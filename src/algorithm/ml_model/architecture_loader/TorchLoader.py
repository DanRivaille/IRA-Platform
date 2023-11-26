from torch.nn import Module

from src.config.ConfigParams import ConfigParams


class TorchLoader:
    def __init__(self, config_params: ConfigParams):
        self.__config = config_params

    def load(self) -> Module:
        pass
