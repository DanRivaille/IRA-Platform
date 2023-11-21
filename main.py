import os

from src.Orchestrator import Orchestrator
from src.config.ConfigParams import ConfigParams
from src.config.CommonPath import CommonPath
from src.algorithm.ml_model.TorchModel import TorchModel
from src.dataset.Z24Dataset import Z24Dataset
from src.utils.ParserArguments import ParserArguments
from src.dataset.preprocessing.Normalizer import Normalizer
from src.dataset.preprocessing.SequenceSplitter import SequenceSplitter


def main():
    args = ParserArguments()

    config_params = ConfigParams.load(os.path.join(CommonPath.CONFIG_FILES_FOLDER.value, args.config_filename))

    preprocess_params = config_params.get_params_dict('preprocess_params')
    sequences_length = preprocess_params.get('sequences_length')
    data_range = (preprocess_params.get('range_lb'), preprocess_params.get('range_up'))

    # Data loading and preprocessing
    normalizer = Normalizer(data_range)
    sequence_splitter = SequenceSplitter(sequences_length)
    preprocessing_steps = [normalizer, sequence_splitter]

    # Model creation
    model = TorchModel.create(config_params, args.model_id)

    orchestrator = Orchestrator(config_params, model, preprocessing_steps)

    if not args.is_test:
        orchestrator.load_train_data(Z24Dataset)
        orchestrator.train_model()

        if args.save:
            orchestrator.save_trained_model()
    else:
        orchestrator.load_test_data(Z24Dataset)
        orchestrator.test_model()

        if args.save:
            orchestrator.save_testing_results()


if __name__ == '__main__':
    main()
