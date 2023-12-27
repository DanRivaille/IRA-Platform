import os

from src.Orchestrator import Orchestrator
from src.algorithm.ml_model.KerasModel import KerasModel
from src.algorithm.ml_model.ModelType import ModelType
from src.algorithm.ml_model.TorchModel import TorchModel
from src.config.CommonPath import CommonPath
from src.config.ConfigParams import ConfigParams
from src.dataset.Z24Dataset import Z24Dataset
from src.dataset.preprocessing.PreprocessStepLoader import PreprocessStepLoader
from src.utils.ParserArguments import ParserArguments
from src.utils.utils import build_model_folderpath


def main():
    args = ParserArguments()

    config_params = ConfigParams.load(os.path.join(CommonPath.CONFIG_FILES_FOLDER.value, args.config_filename))

    preprocess_step_loader = PreprocessStepLoader(config_params)
    preprocessing_steps = preprocess_step_loader.load()

    model_type = ModelType.TORCH_MODEL

    if ModelType.KERAS_MODEL is model_type:
        model_class = KerasModel
    else:
        model_class = TorchModel

    if not args.is_test:
        # Model creation
        model = model_class.create(config_params, args.model_id)

        orchestrator = Orchestrator(config_params, model, preprocessing_steps, args.folder_name)
        orchestrator.load_train_data(Z24Dataset)
        orchestrator.train_model()

        if args.save:
            orchestrator.save_trained_model()
    else:
        # Model creation
        model_folder = build_model_folderpath(args.model_id, config_params.get_params_dict('id'), args.folder_name)
        model_path = 'model_trained' + model_class.get_file_extension()
        model = model_class.load(config_params, args.model_id, os.path.join(model_folder, model_path))

        orchestrator = Orchestrator(config_params, model, preprocessing_steps, args.folder_name)
        orchestrator.load_test_data(Z24Dataset)
        orchestrator.test_model()

        if args.save:
            orchestrator.save_testing_results()


if __name__ == '__main__':
    main()
