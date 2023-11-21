from src.dataset.IRADataset import IRADataset
from src.dataset.preprocessing.PreprocessStep import PreprocessStep


class SequenceSplitter(PreprocessStep):
    def __init__(self, sequence_length):
        self.__sequence_length = sequence_length

    def apply(self, dataset: IRADataset):
        n_samples, sample_length = dataset.data.shape
        sequence_samples_to_consider = (sample_length // self.__sequence_length) * self.__sequence_length

        sequences = dataset.data[:, :sequence_samples_to_consider].reshape((-1, self.__sequence_length))
        dataset.data = sequences
