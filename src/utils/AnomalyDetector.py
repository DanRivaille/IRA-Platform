import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.algorithm.ml_model.TorchModel import TorchModel
from src.config.ConfigParams import ConfigParams


class AnomalyDetector:
    def __init__(self,
                 model: TorchModel,
                 config: ConfigParams):
        self.__trained_model = model
        self.__config = config

        self.__macroseq_length: int = config.get_params_dict('test_params')['macroseq_length']

    def detect_damage(self, damaged_dataloader: DataLoader, healthy_dataloader: DataLoader):
        features_damaged = self.__trained_model.run_test_epoch(damaged_dataloader)
        features_healthy = self.__trained_model.run_test_epoch(healthy_dataloader)

        feature_threshold, macroseq_threshold = self.__find_best_thresholds(features_damaged, features_healthy)

    def __detect_damage(self, feature_vector: np.ndarray, feature_threshold: float,
                        macroseq_threshold: float):
        # Se divide el vector de caracteristicas en macro-secuencias
        macroseq_feature_vector = self.__split_in_macrosequences(feature_vector)

        # Se calcula cuales secuencias dentro de cada macro-secuencia supera el umbral pre-establecido
        labels_vector = AnomalyDetector.__evaluate_thresholds(macroseq_feature_vector, feature_threshold)

        # Se calcula la proporcion de secuencias identificadas como dañadas dentro de cada macro-secuencia
        macrosequences_labels_vector = np.sum(labels_vector, axis=1) / self.__macroseq_length

        # Se etiqueta cada macro-secuencia dependiendo de si supera el umbral pre-establecido
        return AnomalyDetector.__evaluate_thresholds(macrosequences_labels_vector, macroseq_threshold)

    def __split_in_macrosequences(self, labels: np.ndarray) -> np.ndarray:
        n_samples = labels.shape[0]
        samples_to_consider = n_samples - (n_samples % self.__macroseq_length)
        return labels[:samples_to_consider].reshape((-1, self.__macroseq_length))

    @staticmethod
    def __evaluate_thresholds(feature_vector: np.ndarray, threshold: float) -> np.ndarray:
        return (feature_vector > threshold).astype(int)

    def __find_best_thresholds(self, feature_test_vector: np.ndarray, feature_valid_vector: np.ndarray) -> tuple:
        feature_threshold_list = np.linspace(0.0001, 0.05, 10)
        macroseq_threshold_list = np.linspace(0.3, 0.6, 10)

        max_auc = -1
        max_f1 = -1.0
        best_f_t = -1
        best_m_t = -1

        for f_t in feature_threshold_list:
            for m_t in macroseq_threshold_list:
                damage_predicted = self.__detect_damage(feature_test_vector, f_t, m_t)
                health_predicted = self.__detect_damage(feature_valid_vector, f_t, m_t)

                true_labels = (np.concatenate((np.ones(damage_predicted.shape[0], dtype=int),
                                               np.zeros(health_predicted.shape[0], dtype=int)))).reshape((-1, 1))
                predicted_labels = (np.concatenate((damage_predicted, health_predicted))).reshape((-1, 1))

                auc_score = roc_auc_score(true_labels, predicted_labels)
                f1 = f1_score(true_labels, predicted_labels)

                if f1 > max_f1:
                    max_f1 = f1
                    max_auc = auc_score
                    best_f_t = f_t
                    best_m_t = m_t

        print(f'Best values: F_t: {best_f_t} - M_t: {best_m_t} - Max AUC score: {max_auc} - Max F1 score: {max_f1}')
        return best_f_t, best_m_t
