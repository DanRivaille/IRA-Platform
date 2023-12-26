from src.dataset.IRADataset import IRADataset
from src.dataset.preprocessing.PreprocessStep import PreprocessStep


class SequenceSplitter(PreprocessStep):
    """
    A preprocess step that splits the dataset into sequences.
    """
    def __init__(self, sequence_length: int, sequence_features: int = 0):
        """
        Initializes an instance of SequenceNormalizer.
        @param sequence_length he length of each sequence.
        @param sequence_features The number of features in each sequence.
        """
        self.__sequence_length: int = sequence_length
        self.__sequence_features: int = sequence_features

        self.__values_per_sample: int = sequence_length
        if sequence_features != 0:
            self.__sequence_shape: tuple = (sequence_length, sequence_features)
            self.__values_per_sample *= self.__sequence_features
        else:
            self.__sequence_shape: tuple = (sequence_length, )

    def apply(self, dataset: IRADataset):
        """
        Splits the data in the given IRADataset into sequences of the specified length.
        @param dataset A class for loading the datasets used by models.
        """
        n_samples, sample_length = dataset.data.shape
        sequence_samples_to_consider = (sample_length // self.__values_per_sample) * self.__values_per_sample
        sequences = dataset.data[:, :sequence_samples_to_consider].reshape((-1, *self.__sequence_shape))
        dataset.data = sequences

    def save(self, folder: str):
        """
        Saves the preprocess step to a specified folder.
        @param folder The folder path where the preprocess step should be saved.
        """
        pass

    def load(self, folder: str):
        """
        Loads the preprocess step from a specified folder.
        @param folder The folder path from which the preprocess step should be loaded.
        """
        pass

    @staticmethod
    def get_filename() -> str:
        """
        Returns the filename associated with the preprocess step.
        """
        pass
