import numpy as np
import time
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
from src.algorithm.ml_model.architecture_loader.KerasLoader import KerasLoader
from src.algorithm.ml_model.models.AEKeras import get_autoencoder_keras
from src.algorithm.ml_model.models.LSTM_AE_Keras import get_lstm_ae_keras
from src.config.ConfigParams import ConfigParams


class KerasModel(MLModel):
    """
    A class representing a Keras model.
    """
    def __init__(self,
                 identifier: str,
                 learning_rate: float,
                 model_loader: KerasLoader):
        """
        Initializes an instance of KerasModel.
        @param identifier An identifier for the model.
        @param learning_rate The learning rate for the model.
        @param model_loader A class responsible for loading a Keras model based on configuration parameters.
        """
        super().__init__(identifier)
        self.__device = KerasModel.__get_device()

        self.__reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=4, min_lr=learning_rate/100)
        print(self.__device)

        # instanciate a keras model
        with device(self.__device):
            self.model: Model = model_loader.load()
            #self.model: Model = get_autoencoder_keras(1000, Adam(learning_rate=learning_rate), 'mean_squared_error')
            #self.model: Model = get_lstm_ae_keras(50, 1, Adam(learning_rate=learning_rate), 'mean_squared_error')
            print(self.model.summary())

    @staticmethod
    def get_file_extension():
        """
        Get the file extension for the file with the parameters of the Keras model.
        """
        return '.keras'

    @staticmethod
    def get_model_type() -> ModelType:
        """
        Get the Keras model type.
        """
        return ModelType.KERAS_MODEL

    @staticmethod
    def __get_device():
        """
        Gets the device string for TensorFlow operations, either GPU or CPU.
        """
        print(list_physical_devices('GPU'))
        if len(list_physical_devices('GPU')) > 0:
            return '/device:GPU:0'
        else:
            return '/device:CPU:0'

    @staticmethod
    def load(config: ConfigParams, identifier: str, path: str):
        """
        Loads the Keras model with the specified configurations.
        @param config A class for loading configuration parameters related to a model.
        @param identifier An identifier for the model.
        @param path The directory path of the configurations to load.
        """
        keras_model = KerasModel.create(config, identifier)
        keras_model.model = load_model(path)
        return keras_model

    @staticmethod
    def create(config: ConfigParams, identifier: str):
        """
        Creates the Keras model with the specified configurations.
        @param config A class for loading configuration parameters related to a model.
        @param identifier An identifier for the model.
        """
        learning_rate = config.get_params('train_params')['learning_rate']
        model_loader = KerasLoader(config)

        return KerasModel(identifier, learning_rate, model_loader)

    def save(self, config: ConfigParams, path: str):
        """
        Saves the Keras model to the specified path.
        @param path The directory path of the configurations to save.
        """
        self.model.save(path)

    def train(self, config: ConfigParams, trainloader: np.ndarray, validationloader: np.ndarray | None) -> History:
        """
        Runs the train process of the Keras model.
        @param config A class for loading configuration parameters related to a model.
        @param trainloader Array containing training data.
        @param validationloader Array containing validation data.
        @return History A class for saving the training history data of model execution.
        """
        num_epochs = config.get_params('train_params')['num_epochs']
        batch_size = config.get_params('train_params')['batch_size']

        start_time = time.time()

        with device(self.__device):
            history = self.model.fit(
                x=trainloader, y=trainloader,
                validation_data=(validationloader, validationloader),
                epochs=num_epochs,
                batch_size=batch_size,
                callbacks=[self.__reduce_lr],
                verbose=1
            )

        end_time = time.time()
        elapsed_time = end_time - start_time

        history_dict = history.history
        _, train_error_per_sample = self.predict(trainloader, return_per_sample=True)
        learning_rate_uptating = np.array(history_dict['lr']).tolist()
        return History(history_dict['loss'], history_dict['val_loss'], learning_rate_uptating, train_error_per_sample, elapsed_time)

    # TODO: Check this function
    def test(self, config: ConfigParams, testloader: np.ndarray, validationloader: np.ndarray):
        pass

    def predict(self, dataloader: np.ndarray, return_per_sample: bool = True, **kwargs) -> tuple:
        """
        Predict the data in dataloader using the current state of the model.
        @param dataloader Array containing the data to predict.
        @param return_per_sample If it is True, return the error per sample.
        @return Tuple: (predictions, errors per sample)
        """
        sequences_predicted = self.model.predict(dataloader, verbose=0)
        error_per_sample = mean_squared_error(dataloader, sequences_predicted).numpy()

        if return_per_sample:
            return None, error_per_sample
        else:
            return None, error_per_sample.mean()
