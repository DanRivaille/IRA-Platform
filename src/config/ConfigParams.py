from json import load, dump


class ConfigParams:
    def __init__(self, config_dict: dict, original_filepath: str):
        self.params = config_dict
        self.original_filepath = original_filepath

    def save(self, overwrite_original: bool = True, new_path: str = "") -> None:
        if overwrite_original:
            path_to_save = self.original_filepath
        else:
            path_to_save = new_path

        with open(path_to_save, 'w') as config_file:
            dump(self.params, config_file)

    @staticmethod
    def load(filepath: str):
        with open(filepath, 'r') as config_file:
            config_params = load(config_file)

        return ConfigParams(config_params, filepath)
