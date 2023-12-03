from src.dataset.IRADataset import IRADataset
from src.dataset.preprocessing.PreprocessStep import PreprocessStep


class SequenceSplitter(PreprocessStep):
    def __init__(self, sequence_length: int, sequence_features: int = 0):
        self.__sequence_length: int = sequence_length
        self.__sequence_features: int = sequence_features

        self.__values_per_sample: int = sequence_length
        if sequence_features != 0:
            self.__sequence_shape: tuple = (sequence_length, sequence_features)
            self.__values_per_sample *= self.__sequence_features
        else:
            self.__sequence_shape: tuple = (sequence_length, )

    def apply(self, dataset: IRADataset):
        n_samples, sample_length = dataset.data.shape
        sequence_samples_to_consider = (sample_length // self.__values_per_sample) * self.__values_per_sample
        sequences = dataset.data[:, :sequence_samples_to_consider].reshape((-1, *self.__sequence_shape))
        dataset.data = sequences

    def save(self, folder: str):
        pass

    def load(self, folder: str):
        pass

    @staticmethod
    def get_filename() -> str:
        pass
