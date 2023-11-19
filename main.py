from datetime import datetime

from src.dataset.Z24Dataset import Z24Dataset
from src.config.ConfigParams import ConfigParams
from src.utils.plot_functions import plot_signal

config_params = ConfigParams.load('/home/ivan.santos/repositories/IRA-Platform/config_files/config_example.json')

first_date = datetime.strptime(config_params.get_params_dict('train_params').get('first_date'), "%d/%m/%Y")
last_date = datetime.strptime(config_params.get_params_dict('train_params').get('last_date'), "%d/%m/%Y")
sensor_number = config_params.get_params_dict('preprocess_params').get('sensor_number')

sequences_length = config_params.get_params_dict('preprocess_params').get('sequences_length')


dataset = Z24Dataset.load(first_date, last_date, sensor_number)


plot_signal(dataset.data[1][:sequences_length], save=True, filename='original.png')

dataset.normalize_data(is_train_data=True, inplace=True)
dataset.reshape_in_sequences(sequences_length, inplace=True)

plot_signal(dataset.data[65], save=True, filename='reshaped.png')

print(dataset.data.shape)
