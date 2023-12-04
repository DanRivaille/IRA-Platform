from scipy.signal import butter, lfilter

from src.dataset.IRADataset import IRADataset
from src.dataset.preprocessing.PreprocessStep import PreprocessStep


class LowPassFilter(PreprocessStep):
    def __init__(self, cut_frequency: float, sample_rate: float, order: int):
        self.__cut_frequency: float = cut_frequency
        self.__sample_rate: float = sample_rate
        self.__order: int = order

    def apply(self, dataset: IRADataset):
        original_shape = dataset.data.shape
        original_signal = dataset.data.flatten()

        filtered_signal = LowPassFilter.__butter_lowpass_filter(original_signal, self.__cut_frequency,
                                                                self.__sample_rate, self.__order)
        dataset.data = filtered_signal.reshape(original_shape)

    def save(self, folder: str):
        pass

    def load(self, folder: str):
        pass

    @staticmethod
    def get_filename() -> str:
        pass

    @staticmethod
    def __butter_lowpass_filter(original_signal, cutoff, sample_rate, order):
        coeff_b, coeff_a = butter(order, cutoff, fs=sample_rate, btype='low', analog=False)
        filtered_signal = lfilter(coeff_b, coeff_a, original_signal)
        return filtered_signal
