from abc import ABC, abstractmethod

from src.algorithm.Algorithm import Algorithm


class MLModel(Algorithm, ABC):

    """
    Runs the train process
    """
    @abstractmethod
    def train(self):
        pass

    """
    Runs the test process
    """
    @abstractmethod
    def test(self):
        pass
