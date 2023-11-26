import numpy as np
from keras import Model
from keras.models import load_model
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from tensorflow.python.framework.config import list_physical_devices
from tensorflow import device
from keras.callbacks import ReduceLROnPlateau

from src.algorithm.ml_model.History import History
from src.algorithm.ml_model.MLModel import MLModel
from src.algorithm.ml_model.ModelType import ModelType
from src.algorithm.ml_model.models.AEKeras import get_autoencoder_keras
from src.config.ConfigParams import ConfigParams


class KerasModel(MLModel):
    def __init__(self, identifier, input_length: int, learning_rate: float):
        super().__init__(identifier)

        self.__optimizer = Adam(learning_rate=learning_rate)
        self.__loss_function = mean_squared_error
        self.__device = KerasModel.__get_device()

        self.__reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                                             patience=4, min_lr=learning_rate / 100)
        print(self.__device)

        # instanciate a keras model
        with device(self.__device):
            self.model: Model = get_autoencoder_keras(input_length, self.__optimizer, self.__loss_function)

    @staticmethod
    def get_file_extension():
        return '.keras'

    @staticmethod
    def get_model_type() -> ModelType:
        return ModelType.KERAS_MODEL

    @staticmethod
    def __get_device():
        print(list_physical_devices('GPU'))
        if len(list_physical_devices('GPU')) > 0:
            return '/device:GPU:0'
        else:
            return '/device:CPU:0'

    @staticmethod
    def load(config: ConfigParams, identifier: str, path: str):
        keras_model = KerasModel.create(config, identifier)
        keras_model.model = load_model(path)
        return keras_model

    @staticmethod
    def create(config: ConfigParams, identifier: str):
        sequences_length = config.get_params_dict('preprocess_params')['sequences_length']
        learning_rate = config.get_params_dict('train_params')['learning_rate']
        return KerasModel(identifier, sequences_length, learning_rate)

    def save(self, config: ConfigParams, path: str):
        self.model.save(path)

    def train(self, config: ConfigParams, trainloader: np.ndarray, validationloader: np.ndarray | None) -> History:
        num_epochs = config.get_params_dict('train_params')['num_epochs']
        batch_size = config.get_params_dict('train_params')['batch_size']

        with device(self.__device):
            history = self.model.fit(
                x=trainloader, y=trainloader,
                validation_data=(validationloader, validationloader),
                epochs=num_epochs,
                batch_size=batch_size,
                callbacks=[self.__reduce_lr],
                verbose=1
            )

        history_dict = history.history
        _, train_error_per_sample = self.predict(trainloader, return_per_sample=True)
        learning_rate_uptating = np.array(history_dict['lr']).tolist()
        return History(history_dict['loss'], history_dict['val_loss'], learning_rate_uptating, train_error_per_sample)
        pass

    # TODO: Check this function
    def test(self, config: ConfigParams, testloader: np.ndarray, validationloader: np.ndarray):
        pass

    def predict(self, dataloader: np.ndarray, return_per_sample: bool = True, **kwargs) -> tuple:
        sequences_predicted = self.model.predict(dataloader, verbose=0)
        error_per_sample = mean_squared_error(dataloader, sequences_predicted).numpy()

        if return_per_sample:
            return None, error_per_sample
        else:
            return None, error_per_sample.mean()
