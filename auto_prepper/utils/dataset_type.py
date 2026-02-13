from enum import Enum


class DatasetType(Enum):
    TRAIN = 0
    TEST = 1
    INFERENCE = 2

    def is_within_threshold(self, dataset_type_threshold):
        return self.value <= dataset_type_threshold.value

    @classmethod
    def from_str(cls, s):
        if s == 'inference':
            return DatasetType.INFERENCE
        if s == 'test':
            return DatasetType.TEST
        if s == 'train':
            return DatasetType.TRAIN
        return None

    def __str__(self):
        if self == DatasetType.INFERENCE:
            return 'inference'
        if self == DatasetType.TEST:
            return 'test'
        return 'train'

    def __repr__(self):
        return self.__str__()
