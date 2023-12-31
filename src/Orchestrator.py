import os

from src.algorithm.Results import Results
from src.algorithm.ml_model.History import History
from src.algorithm.ml_model.MLModel import MLModel
from src.config.ConfigParams import ConfigParams
from src.dataset.IRADataset import IRADataset
from src.dataset.dataset_type import DatasetType
from src.dataset.preprocessing.PreprocessStep import PreprocessStep
from src.domain.AnomalyDetector import AnomalyDetector
from src.utils.utils import build_model_folderpath


class Orchestrator:
    """
    Class manages the overall process of loading, preprocessing, training, testing, and saving models.
    """
    def __init__(self, config_params: ConfigParams,
                 model: MLModel,
                 preprocessing_steps: [PreprocessStep],
                 folder_name: str):
        """
        Initializes an instance of Orchestrator.
        @param config_params A class for loading and saving configuration parameters related to a model.
        @param model The current machine learning model.
        @param preprocessing_steps The list of preprocessing steps to be applied to the data.
        @param folder_name The folder name to save the runs.
        """
        self.__results: Results | None = None
        self.__config_params: ConfigParams = config_params
        self.__model: MLModel = model
        self.__preprocessing_steps: [PreprocessStep] = preprocessing_steps

        self.__train_dataset: IRADataset | None = None
        self.__test_dataset: IRADataset | None = None
        self.__valid_dataset: IRADataset | None = None

        self.__history: History | None = None

        self.__model_folder: str = build_model_folderpath(self.__model.identifier,
                                                          self.__config_params.get_params('id'),
                                                          folder_name)

    def load_train_data(self, class_dataset: IRADataset.__class__):
        """
        Loads the training and validation datasets and applies preprocessing steps.
        @param class_dataset The class of the dataset to be loaded.
        """
        self.__train_dataset = class_dataset.load(self.__config_params, DatasetType.TRAIN_DATA)
        self.__valid_dataset = class_dataset.load(self.__config_params, DatasetType.VALIDATION_TRAIN_DATA)

        self.__preprocess_data()

    def load_test_data(self, class_dataset: IRADataset.__class__):
        """
        Loads the testing and validation datasets and applies preprocessing steps.
        @param class_dataset The class of the dataset to be loaded.
        """
        self.__test_dataset = class_dataset.load(self.__config_params, DatasetType.TEST_DATA)
        self.__valid_dataset = class_dataset.load(self.__config_params, DatasetType.VALIDATION_TEST_DATA)

        for preprocess_step in self.__preprocessing_steps:
            preprocess_step.load(self.__model_folder)

        self.__preprocess_data()

    def __preprocess_data(self):
        """
        Applies the list of preprocessings steps to the datasets.
        """
        for preprocess_step in self.__preprocessing_steps:
            Orchestrator.__apply_preprocess_step(self.__train_dataset, preprocess_step)
            Orchestrator.__apply_preprocess_step(self.__test_dataset, preprocess_step)
            Orchestrator.__apply_preprocess_step(self.__valid_dataset, preprocess_step)

    @staticmethod
    def __apply_preprocess_step(dataset: IRADataset, preprocess_step: PreprocessStep):
        """
        Applies a single preprocessing step to the given dataset.
        @param dataset The dataset to be preprocessed.
        @param preprocess_step The preprocessing step to be applied.
        """
        if dataset is not None:
            preprocess_step.apply(dataset)

    def train_model(self):
        """
        Trains the machine learning model using the training and validation datasets.
        """
        batch_size = self.__config_params.get_params('train_params')['batch_size']
        train_loader = self.__train_dataset.get_dataloader(self.__model.get_model_type(), batch_size)
        valid_loader = self.__valid_dataset.get_dataloader(self.__model.get_model_type(), batch_size)

        # Model training
        self.__history = self.__model.train(self.__config_params, train_loader, valid_loader)

    def test_model(self):
        """
        Tests the machine learning model using the testing and validation datasets.
        """
        batch_size = self.__config_params.get_params('train_params')['batch_size']
        test_loader = self.__test_dataset.get_dataloader(self.__model.get_model_type(), batch_size)
        valid_loader = self.__valid_dataset.get_dataloader(self.__model.get_model_type(), batch_size)

        anomaly_detector = AnomalyDetector(self.__model, self.__config_params)
        self.__results = anomaly_detector.detect_damage(test_loader, valid_loader)

    def save_trained_model(self):
        """
        Saves the trained model, training history, and preprocessing steps.
        """
        os.makedirs(self.__model_folder, exist_ok=True)

        model_path = os.path.join(self.__model_folder, 'model_trained' + self.__model.get_file_extension())
        results_path = os.path.join(self.__model_folder, 'history.json')

        self.__history.save(results_path)
        self.__model.save(self.__config_params, model_path)

        for preprocess_step in self.__preprocessing_steps:
            preprocess_step.save(self.__model_folder)

    def save_testing_results(self):
        """
        Saves the results of the anomaly detection process.
        """
        results_path = os.path.join(self.__model_folder, 'results.json')
        self.__results.save(results_path)
