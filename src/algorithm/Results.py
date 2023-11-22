from json import dump

import numpy as np


class Results:
    def __init__(self,
                 feature_threshold: float,
                 macroseq_threshold: float,
                 max_f1: float,
                 max_auc: float,
                 damaged_features: np.ndarray,
                 healthy_features: np.ndarray):
        self.__feature_threshold = feature_threshold
        self.__macroseq_threshold = macroseq_threshold
        self.__max_f1 = max_f1
        self.__max_auc = max_auc
        self.__damaged_features = damaged_features
        self.__healthy_features = healthy_features

    def __get_json_dict(self) -> dict:
        return {
            "feature_threshold": self.__feature_threshold,
            "macroseq_threshold": self.__macroseq_threshold,
            "max_f1": self.__max_f1,
            "max_auc": self.__max_auc,
            "damaged_features": self.__damaged_features.tolist(),
            "healthy_features": self.__healthy_features.tolist()
        }

    def save(self, path: str):
        with open(path, "w") as json_results_file:
            dump(self.__get_json_dict(), json_results_file)
