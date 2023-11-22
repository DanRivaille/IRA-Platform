import numpy as np
from torch import no_grad, cuda, save, load
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score

from src.algorithm.ml_model.MLModel import MLModel
from src.algorithm.ml_model.models.Autoencoder import Autoencoder
from src.config.ConfigParams import ConfigParams
from src.algorithm.ml_model.History import History


class TorchModel(MLModel):

    def __init__(self, identifier: str, input_length: int, learning_rate: float):
        super().__init__(identifier)

        self.device = TorchModel.__get_device()

        self.model = Autoencoder(input_length)
        self.model.to(self.device)

        self.criterion_train = MSELoss()
        self.criterion_test = MSELoss(reduction='none')
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

    @staticmethod
    def get_file_extension():
        return '.pth'

    @staticmethod
    def __get_device():
        if cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device

    @staticmethod
    def load(config: ConfigParams, identifier: str, path: str):
        torch_model = TorchModel.create(config, identifier)
        torch_model.model.load_state_dict(load(path))
        torch_model.model.eval()
        torch_model.model.to(torch_model.device)

        return torch_model

    @staticmethod
    def create(config: ConfigParams, identifier: str):
        sequences_length = config.get_params_dict('preprocess_params')['sequences_length']
        learning_rate = config.get_params_dict('train_params')['learning_rate']
        return TorchModel(identifier, sequences_length, learning_rate)

    def save(self, config: ConfigParams, path: str):
        save(self.model.state_dict(), path)

    def train(self, config: ConfigParams, trainloader: DataLoader, validationloader: DataLoader | None) -> History:
        num_epochs = config.get_params_dict('train_params')['num_epochs']

        train_error = []
        validation_error = []
        learning_rate_updating = []

        for epoch in range(num_epochs):
            loss = self.__run_epoch(trainloader)
            train_error.append(loss)

            if validationloader is not None:
                validation_loss = self.__run_epoch(validationloader, is_train=False)
                validation_error.append(validation_loss)

            if (epoch % 5) == 0:
                print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss: .7f}')

            learning_rate_updating.append(config.get_params_dict('train_params')['learning_rate'])

        return History(train_error, validation_error, learning_rate_updating)

    def __run_epoch(self, dataloader: DataLoader, is_train=True) -> float:
        loss = None

        for trainbatch in dataloader:
            batch = trainbatch.to(self.device)

            output = self.model(batch)
            loss = self.criterion_train(output, batch.data)

            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    # TODO: Check this function
    def test(self, config: ConfigParams, testloader: DataLoader, validationloader: DataLoader):
        feature_test_vector = self.__run_test(testloader)
        feature_valid_vector = self.__run_test(validationloader)

        macroseq_length = config.get_params_dict('test_params')['macroseq_length']

        feature_threshold, macroseq_threshold = TorchModel.__find_best_thresholds(macroseq_length, feature_test_vector,
                                                                                  feature_valid_vector)

    def __run_test(self, dataset: DataLoader) -> np.ndarray:
        feature_vector = []

        with no_grad():
            for databatch in dataset:
                signals = databatch.to(self.device)
                output = self.model(signals)

                current_feature_value = np.mean(self.criterion_test(output, signals.data).cpu().numpy(), axis=1)
                feature_vector.append(current_feature_value)

        return np.concatenate(feature_vector).flatten()

    @staticmethod
    def __detect_damage(feature_vector: np.ndarray, macrosequences_length: int, feature_threshold: float,
                        macroseq_threshold: float):
        n_samples = feature_vector.shape[0]

        samples_to_consider = n_samples - (n_samples % macrosequences_length)

        # Se divide el vector de caracteristicas en macro-secuencias
        macroseq_feature_vector = TorchModel.__split_in_macrosequences(feature_vector[:samples_to_consider],
                                                                       macrosequences_length)

        # Se calcula cuales secuencias dentro de cada macro-secuencia supera el umbral pre-establecido
        labels_vector = (macroseq_feature_vector > feature_threshold).astype(int)

        # Se calcula la proporcion de secuencias identificadas como dañadas dentro de cada macro-secuencia
        macrosequences_labels_vector = np.sum(labels_vector, axis=1) / macrosequences_length

        # Se etiqueta cada macro-secuencia dependiendo de si supera el umbral pre-establecido
        damaged_macrosequences = (macrosequences_labels_vector > macroseq_threshold).astype(int)
        return damaged_macrosequences

    @staticmethod
    def __split_in_macrosequences(labels, macro_length):
        return labels.reshape((-1, macro_length))

    @staticmethod
    def __find_best_thresholds(macrosequences_length: int, feature_test_vector: np.ndarray,
                               feature_valid_vector: np.ndarray) -> tuple:
        feature_threshold_list = np.linspace(0.0001, 0.05, 100)
        macroseq_threshold_list = np.linspace(0.3, 0.6, 10)

        max_auc = -1
        max_f1 = -1.0
        best_f_t = -1
        best_m_t = -1

        for f_t in feature_threshold_list:
            for m_t in macroseq_threshold_list:
                damage_predicted = TorchModel.__detect_damage(feature_test_vector, macrosequences_length, f_t, m_t)
                health_predicted = TorchModel.__detect_damage(feature_valid_vector, macrosequences_length, f_t, m_t)

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
