from abc import ABC, abstractmethod

from src.algorithm.Algorithm import Algorithm


class MLModel(Algorithm, ABC):

    def __init__(self, identifier: str):
        super(Algorithm).__init__(identifier)

    """
    Runs the train process
    """
    @abstractmethod
    def train(self, config: dict) -> dict:
        pass

    """
    Runs the test process
    """
    @abstractmethod
    def test(self, config: dict) -> dict:
        pass
