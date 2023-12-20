from json import load, dump


class ConfigParams:
    """
    A class for loading and saving configuration parameters related to a model.
    """
    def __init__(self, config_dict: dict, original_filepath: str):
        """
        Initializes an instance of ConfigParams.
        @param config_dict The dictionary containing configuration parameters.
        @param original_filepath The file path containing those parameters.
        """
        self.params = config_dict
        self.original_filepath = original_filepath

    def save(self, overwrite_original: bool = True, new_path: str = "") -> None:
        """
        Saves the configuration parameters to a file.
        @param overwrite_original A flag that, if True overwrites the original file path with the configuration parameters; otherwise, they are saved to a new path. Default is True.
        @param new_path The path to save the configuration parameters if not overwriting the original. Default is "".
        """
        if overwrite_original:
            path_to_save = self.original_filepath
        else:
            path_to_save = new_path

        with open(path_to_save, 'w') as config_file:
            dump(self.params, config_file)

    @staticmethod
    def load(filepath: str):
        """
        Loads the configurations parameters from a file.
        @param filepath The path to the configuration file.
        """
        with open(filepath, 'r') as config_file:
            config_params = load(config_file)

        return ConfigParams(config_params, filepath)

    def get_params_dict(self, key_param: str) -> str | dict | None:
        """
        Returns the specific configuration parameter from the dictionary for the given key.
        @param key_param The key of the parameter to retrieve.
        """
        return self.params.get(key_param, None)
