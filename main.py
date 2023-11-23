import os

from src.Orchestrator import Orchestrator
from src.algorithm.ml_model.TorchModel import TorchModel
from src.config.CommonPath import CommonPath
from src.config.ConfigParams import ConfigParams
from src.dataset.Z24Dataset import Z24Dataset
from src.dataset.preprocessing.SequenceNormalizer import SequenceNormalizer
from src.dataset.preprocessing.SequenceSplitter import SequenceSplitter
from src.utils.ParserArguments import ParserArguments
from src.utils.utils import build_model_folderpath


def main():
    args = ParserArguments()

    config_params = ConfigParams.load(os.path.join(CommonPath.CONFIG_FILES_FOLDER.value, args.config_filename))

    preprocess_params = config_params.get_params_dict('preprocess_params')
    sequences_length = preprocess_params.get('sequences_length')
    data_range = (preprocess_params.get('range_lb'), preprocess_params.get('range_up'))

    # Data loading and preprocessing
    sequence_splitter = SequenceSplitter(sequences_length)
    sequence_normalizer = SequenceNormalizer(data_range)
    preprocessing_steps = [sequence_splitter, sequence_normalizer]

    model_class = TorchModel

    if not args.is_test:
        # Model creation
        model = model_class.create(config_params, args.model_id)

        orchestrator = Orchestrator(config_params, model, preprocessing_steps)
        orchestrator.load_train_data(Z24Dataset)
        orchestrator.train_model()

        if args.save:
            orchestrator.save_trained_model()
    else:
        # Model creation
        model_folder = build_model_folderpath(args.model_id, config_params.get_params_dict('id'))
        model_path = 'model_trained' + model_class.get_file_extension()
        model = model_class.load(config_params, args.model_id, os.path.join(model_folder, model_path))

        orchestrator = Orchestrator(config_params, model, preprocessing_steps)
        orchestrator.load_test_data(Z24Dataset)
        orchestrator.test_model()

        if args.save:
            orchestrator.save_testing_results()


if __name__ == '__main__':
    main()
