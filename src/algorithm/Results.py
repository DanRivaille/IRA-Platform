from json import dump

import numpy as np


class Results:
    """
    A class for saving the results of model execution.
    """
    def __init__(self,
                 feature_threshold: float,
                 macroseq_threshold: float,
                 max_f1: float,
                 max_auc: float,
                 damaged_features: np.ndarray,
                 healthy_features: np.ndarray,
                 execution_time: float):
        """
        Initializes an instance of Results.
        @param feature_threshold The threshold to consider whether a feature is damaged.
        @param macroseq_threshold The threshold to consider whether the macrosequence is damaged.
        @param max_f1 The maximum F1 score achieved.
        @param max_auc The maximum Area Under the Curve (AUC) achieved.
        @param damaged_features An array with the features considered as damaged.
        @param healthy_features An array with the features considered as healthy.
        @param execution_time The execution time of the model's anomaly detection process.
        """
        self.__feature_threshold = feature_threshold
        self.__macroseq_threshold = macroseq_threshold
        self.__max_f1 = max_f1
        self.__max_auc = max_auc
        self.__damaged_features = damaged_features
        self.__healthy_features = healthy_features
        self.__execution_time = execution_time

    def __get_json_dict(self) -> dict:
        """
        Returns a dictionary for a .json file with the results. 
        """
        return {
            "feature_threshold": self.__feature_threshold,
            "macroseq_threshold": self.__macroseq_threshold,
            "max_f1": self.__max_f1,
            "max_auc": self.__max_auc,
            "damaged_features": self.__damaged_features.tolist(),
            "healthy_features": self.__healthy_features.tolist(),
            "execution_time": self.__execution_time
        }

    def save(self, path: str):
        """
        Saves the results to a .json file.
        @param path The file path where the results will be saved.
        """
        with open(path, "w") as json_results_file:
            dump(self.__get_json_dict(), json_results_file)
