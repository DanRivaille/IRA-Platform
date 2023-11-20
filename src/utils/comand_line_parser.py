import argparse


def parse_comand_line_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--id", help="Model identifier")
    parser.add_argument("-c", "--config", help="Configuration file")

    args = parser.parse_args()
    return args
