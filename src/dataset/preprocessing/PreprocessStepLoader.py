from src.dataset.preprocessing.PreprocessStep import PreprocessStep
from src.dataset.preprocessing.Normalizer import Normalizer
from src.dataset.preprocessing.LowPassFilter import LowPassFilter
from src.dataset.preprocessing.SequenceNormalizer import SequenceNormalizer
from src.dataset.preprocessing.SequenceSplitter import SequenceSplitter

from src.config.ConfigParams import ConfigParams


class PreprocessStepLoader:
    """
    The class is responsible for loading and initializing preprocessing steps based on
    the configuration parameters.
    """
    def __init__(self, config_params: ConfigParams):
        """
        Initializes an instance of PreprocessStepLoader.
        @param config_params The configuration parameters for preprocessing steps.
        """
        self.config_params = config_params

    def load(self) -> list:
        """
        Loads and initializes a list of preprocessing steps based on the configuration parameters.
        """
        preprocessing_steps_info = self.config_params.get_params_dict("preprocessing_steps")
        preprocessing_steps = []
       
        for preprocess_step in preprocessing_steps_info:
            preprocessing_steps.append(self.__load_preprocess_step(preprocess_step))
        
        return preprocessing_steps

    def __load_preprocess_step(self, preprocess_step_info: dict ) -> PreprocessStep:
        """
        Loads and initializes a single preprocessing step based on the provided information.
        @param preprocess_step_info A dictionary with information about the preprocessing step.
        """
        preprocess_params = preprocess_step_info.get('params')
        preprocess_type = preprocess_step_info.get('type')

        if preprocess_type == "low_pass_filter":
            return LowPassFilter(preprocess_params.get("cut_frequency"), 
                                 preprocess_params.get("sample_rate"), 
                                 preprocess_params.get("order"))

        elif preprocess_type == "normalizer":
            return Normalizer((preprocess_params.get("range")[0], 
                               preprocess_params.get("range")[1]))

        elif preprocess_type == "sequence_normalizer":
            return SequenceNormalizer((preprocess_params.get("range")[0],
                                       preprocess_params.get("range")[1]))

        elif preprocess_type == "sequence_splitter":
            sequence_features = preprocess_params.get("sequence_features")
            if sequence_features is not None:
                return SequenceSplitter(preprocess_params.get("sequences_length"), sequence_features)
            else:
                return SequenceSplitter(preprocess_params.get("sequences_length"))

