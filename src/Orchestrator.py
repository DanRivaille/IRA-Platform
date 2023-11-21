import os

from torch.utils.data import DataLoader

from src.algorithm.ml_model.History import History
from src.algorithm.ml_model.MLModel import MLModel
from src.config.CommonPath import CommonPath
from src.config.ConfigParams import ConfigParams
from src.dataset.IRADataset import IRADataset
from src.dataset.dataset_type import DatasetType
from src.dataset.preprocessing.PreprocessStep import PreprocessStep
from src.utils.utils import build_model_folderpath


class Orchestrator:
    def __init__(self, config_params: ConfigParams,
                 model: MLModel,
                 preprocessing_steps: [PreprocessStep]):
        self.__config_params = config_params
        self.__model = model
        self.__preprocessing_steps = preprocessing_steps

        self.__train_dataset: IRADataset | None = None
        self.__test_dataset: IRADataset | None = None
        self.__valid_dataset: IRADataset | None = None

        self.__history: History | None = None

    def load_train_data(self, class_dataset: IRADataset.__class__):
        self.__train_dataset = class_dataset.load(self.__config_params, DatasetType.TRAIN_DATA)
        self.__valid_dataset = class_dataset.load(self.__config_params, DatasetType.VALIDATION_TRAIN_DATA)

        for preprocess_step in self.__preprocessing_steps:
            preprocess_step.apply(self.__train_dataset)
            preprocess_step.apply(self.__valid_dataset)

    def load_test_data(self, class_dataset: IRADataset.__class__):
        self.__test_dataset = class_dataset.load(self.__config_params, DatasetType.TEST_DATA)
        pass

    def train_model(self):
        batch_size = self.__config_params.get_params_dict('train_params')['batch_size']
        train_loader = DataLoader(self.__train_dataset.get_torch_dataset(), batch_size=batch_size)
        valid_loader = DataLoader(self.__valid_dataset.get_torch_dataset(), batch_size=batch_size)

        # Model training
        self.__history = self.__model.train(self.__config_params, train_loader, valid_loader)

    def test_model(self):
        print('Testing the model')
        pass

    def save_trained_model(self):
        output_model_folder = build_model_folderpath(self.__model.identifier,
                                                     self.__config_params.get_params_dict('id'))
        os.makedirs(output_model_folder, exist_ok=True)

        model_path = os.path.join(output_model_folder, 'model_trained' + self.__model.get_file_extension())
        results_path = os.path.join(output_model_folder, 'history.json')

        self.__history.save(results_path)
        self.__model.save(self.__config_params, model_path)

    def save_testing_results(self):
        print('Saving the results')
        pass
