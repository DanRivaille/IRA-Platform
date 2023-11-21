from src.dataset.dataset_type import DatasetType
from src.dataset.preprocessing.PreprocessStep import PreprocessStep


class SequenceSplitter(PreprocessStep):
    def __init__(self, sequence_length):
        self.__sequence_length = sequence_length

    def apply(self, data, data_type: DatasetType | None = None):
        n_samples, sample_length = data.shape

        sequence_samples_to_consider = (sample_length // self.__sequence_length) * self.__sequence_length
        return data[:, :sequence_samples_to_consider].reshape((-1, self.__sequence_length))
