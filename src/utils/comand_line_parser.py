import os
import argparse


class Parser:
    def __init__(self):
        self.__parser = argparse.ArgumentParser()
        self.__create_arguments()
        self.__args = self.__parser.parse_args()

    def __create_arguments(self):
        self.__parser.add_argument("-i", "--id", help="Model identifier", required=True)
        self.__parser.add_argument("-c", "--config", help="Configuration file", required=True)
        self.__parser.add_argument("-s", "--save", help="Save the trained model and its results", default=False)

    @property
    def model_id(self):
        return self.__args.id

    @property
    def config_filename(self):
        return os.path.split(self.__args.config)[1]

    @property
    def save(self):
        return self.__args.save
