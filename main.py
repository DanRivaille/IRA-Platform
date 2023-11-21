import os

from torch.utils.data import DataLoader

from src.dataset.Z24Dataset import Z24Dataset
from src.config.ConfigParams import ConfigParams
from src.config.CommonPath import CommonPath
from src.algorithm.ml_model.TorchModel import TorchModel
from src.dataset.dataset_type import DatasetType
from src.utils.ParserArguments import ParserArguments
from src.dataset.preprocessing.Normalizer import Normalizer
from src.dataset.preprocessing.SequenceSplitter import SequenceSplitter


def main():
    args = ParserArguments()

    config_params = ConfigParams.load(os.path.join(CommonPath.CONFIG_FILES_FOLDER.value, args.config_filename))

    sequences_length = config_params.get_params_dict('preprocess_params').get('sequences_length')
    batch_size = config_params.get_params_dict('train_params')['batch_size']

    # Data loading and preprocessing
    normalizer = Normalizer((-1, 1))
    sequence_splitter = SequenceSplitter(sequences_length)
    preprocessing_steps = [normalizer, sequence_splitter]

    train_dataset = Z24Dataset.load(config_params, DatasetType.TRAIN_DATA)
    valid_dataset = Z24Dataset.load(config_params, DatasetType.VALIDATION_DATA)

    for preprocess_step in preprocessing_steps:
        preprocess_step.apply(train_dataset)
        preprocess_step.apply(valid_dataset)

    train_loader = DataLoader(train_dataset.get_torch_dataset(), batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset.get_torch_dataset(), batch_size=batch_size)

    # Model creation
    model = TorchModel.create(config_params, args.model_id)

    # Model training
    history = model.train(config_params, train_loader, valid_loader)

    # Saving the results
    if args.save:

        output_model_folder = os.path.join(CommonPath.MODEL_PARAMETERS_FOLDER.value, model.idenfitier)
        os.makedirs(output_model_folder, exist_ok=True)

        history.save(output_model_folder)
        model.save(config_params, output_model_folder)


if __name__ == '__main__':
    main()
