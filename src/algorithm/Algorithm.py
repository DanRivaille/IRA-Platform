from abc import ABC, abstractmethod

from src.config.ConfigParams import ConfigParams


class Algorithm(ABC):

    def __init__(self, identifier: str):
        self.identifier = identifier

    """
    Creates a new instance of the algorithm
    """
    @staticmethod
    @abstractmethod
    def create(config: ConfigParams, identifier: str):
        pass

    """
    Loads an algorithm instance using a file
    """
    @staticmethod
    @abstractmethod
    def load(config: ConfigParams, path: str):
        pass

    """
    Saves the current state of the algorithm
    """
    @abstractmethod
    def save(self, config: ConfigParams, path: str):
        pass
