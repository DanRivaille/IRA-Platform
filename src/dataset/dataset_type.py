from enum import Enum


class DatasetType(Enum):
    TRAIN_DATA = 'train_params'
    TEST_DATA = 'test_params'
    VALIDATION_DATA = 'validation_params'
