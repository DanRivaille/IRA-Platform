from abc import ABC, abstractmethod


class Algorithm(ABC):

    def __init__(self, identifier: str):
        self.identifier = identifier

    """
    Creates a new instance of the algorithm
    """
    @staticmethod
    @abstractmethod
    def create(config: dict):
        pass

    """
    Loads an algorithm instance using a file
    """
    @staticmethod
    @abstractmethod
    def load(config: dict):
        pass

    """
    Saves the current state of the algorithm
    """
    @abstractmethod
    def save(self, config: dict) -> bool:
        pass
