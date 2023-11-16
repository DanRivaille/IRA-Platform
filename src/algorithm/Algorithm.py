from abc import ABC, abstractmethod


class Algorithm(ABC):
    """
    Creates a new instance of the algorithm
    """
    @staticmethod
    @abstractmethod
    def create():
        pass

    """
    Loads an algorithm instance using a file
    """
    @staticmethod
    @abstractmethod
    def load():
        pass

    """
    Saves the current state of the algorithm
    """
    @staticmethod
    @abstractmethod
    def save():
        pass
