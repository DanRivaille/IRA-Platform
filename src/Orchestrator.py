import os

from torch.utils.data import DataLoader

from src.algorithm.ml_model.History import History
from src.algorithm.ml_model.TorchModel import TorchModel
from src.config.CommonPath import CommonPath
from src.config.ConfigParams import ConfigParams
from src.dataset.IRADataset import IRADataset
from src.dataset.Z24Dataset import Z24Dataset
from src.dataset.dataset_type import DatasetType
from src.dataset.preprocessing.PreprocessStep import PreprocessStep


class Orchestrator:
    def __init__(self, config_params: ConfigParams,
                 model: TorchModel,
                 preprocessing_steps: [PreprocessStep]):
        self.__config_params = config_params
        self.__model = model
        self.__preprocessing_steps = preprocessing_steps

        self.__train_dataset: IRADataset | None = None
        self.__test_dataset: IRADataset | None = None
        self.__valid_dataset: IRADataset | None = None

        self.__history: History | None = None

    def load_train_data(self):
        self.__train_dataset = Z24Dataset.load(self.__config_params, DatasetType.TRAIN_DATA)
        self.__valid_dataset = Z24Dataset.load(self.__config_params, DatasetType.VALIDATION_DATA)

        for preprocess_step in self.__preprocessing_steps:
            preprocess_step.apply(self.__train_dataset)
            preprocess_step.apply(self.__valid_dataset)

    def train_model(self):
        batch_size = self.__config_params.get_params_dict('train_params')['batch_size']
        train_loader = DataLoader(self.__train_dataset.get_torch_dataset(), batch_size=batch_size)
        valid_loader = DataLoader(self.__valid_dataset.get_torch_dataset(), batch_size=batch_size)

        # Model training
        self.__history = self.__model.train(self.__config_params, train_loader, valid_loader)

    def save_model(self):
        model_folder = f'{self.__model.idenfitier}_cnf_{self.__config_params.get_params_dict("id")}'
        output_model_folder = os.path.join(CommonPath.MODEL_PARAMETERS_FOLDER.value, model_folder)
        os.makedirs(output_model_folder, exist_ok=True)

        model_path = os.path.join(output_model_folder, 'model_trained' + TorchModel.get_file_extension())
        results_path = os.path.join(output_model_folder, 'history.json')

        self.__history.save(results_path)
        self.__model.save(self.__config_params, model_path)
