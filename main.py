import os

from torch.utils.data import DataLoader

from src.dataset.Z24Dataset import Z24Dataset
from src.config.ConfigParams import ConfigParams
from src.config.CommonPath import CommonPath
from src.algorithm.ml_model.TorchModel import TorchModel
from src.dataset.dataset_type import DatasetType
from src.utils.comand_line_parser import parse_comand_line_arguments


def main():
    args = parse_comand_line_arguments()
    config_filename = args.config
    model_identifier = args.id

    config_params = ConfigParams.load(os.path.join(CommonPath.CONFIG_FILES_FOLDER.value, config_filename))

    sequences_length = config_params.get_params_dict('preprocess_params').get('sequences_length')

    train_dataset = Z24Dataset.load(config_params, DatasetType.TRAIN_DATA)
    valid_dataset = Z24Dataset.load(config_params, DatasetType.VALIDATION_DATA)

    train_dataset.normalize_data((-1, 1), inplace=True)
    train_dataset.reshape_in_sequences(sequences_length, inplace=True)

    valid_dataset.normalize_data((-1, 1), inplace=True)
    valid_dataset.reshape_in_sequences(sequences_length, inplace=True)

    batch_size = config_params.get_params_dict('train_params')['batch_size']
    train_loader = DataLoader(train_dataset.get_torch_dataset(), batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset.get_torch_dataset(), batch_size=batch_size)

    model = TorchModel.create(config_params, model_identifier)

    history = model.train(config_params, train_loader, valid_loader)

    output_model_folder = os.path.join(CommonPath.MODEL_PARAMETERS_FOLDER.value, model.idenfitier)
    os.makedirs(output_model_folder, exist_ok=True)

    history.save(output_model_folder)
    model.save(config_params, output_model_folder)


if __name__ == '__main__':
    main()
