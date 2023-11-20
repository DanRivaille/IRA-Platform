import argparse


class Parser:
    def __init__(self):
        self.__parser = argparse.ArgumentParser()
        self.__create_arguments()
        self.__args = self.__parser.parse_args()

    def __create_arguments(self):
        self.__parser.add_argument("-i", "--id", help="Model identifier")
        self.__parser.add_argument("-c", "--config", help="Configuration file")

    @property
    def model_id(self):
        return self.__args.id

    @property
    def config_filename(self):
        return self.__args.config
