from scipy.signal import butter, lfilter

from src.dataset.IRADataset import IRADataset
from src.dataset.preprocessing.PreprocessStep import PreprocessStep


class LowPassFilter(PreprocessStep):
    """
    A preprocess step that applies a low-pass filter to the data in an IRADataset.
    """
    def __init__(self, cut_frequency: float, sample_rate: float, order: int):
        """
        Initializes an instance of LowPassFilter.
        @param cut_frequency The cutoff frequency for the low-pass filter.
        @param sample_rate The sample rate of the data.
        @param order The order of the low-pass filter.
        """
        self.__cut_frequency: float = cut_frequency
        self.__sample_rate: float = sample_rate
        self.__order: int = order

    def apply(self, dataset: IRADataset):
        """
        Applies the low-pass filter to the data in the given IRADataset.
        @param dataset A class for loading the datasets used by models.
        """
        original_shape = dataset.data.shape
        original_signal = dataset.data.flatten()

        filtered_signal = LowPassFilter.__butter_lowpass_filter(original_signal, self.__cut_frequency,
                                                                self.__sample_rate, self.__order)
        dataset.data = filtered_signal.reshape(original_shape)

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

    @staticmethod
    def __butter_lowpass_filter(original_signal, cutoff, sample_rate, order):
        """
        Applies a Butterworth low-pass filter to the provided signal.
        @param original_signal The input signal.
        @param cutoff The cutoff frequency of the filter.
        @param sample_rate The sample rate of the signal.
        @param order The order of the filter.
        """
        coeff_b, coeff_a = butter(order, cutoff, fs=sample_rate, btype='low', analog=False)
        filtered_signal = lfilter(coeff_b, coeff_a, original_signal)
        return filtered_signal
