from enum import Enum


def process_enums_list(enum_list):
    if not all(isinstance(s, str) for s in enum_list):
        enum_list = [s.value for s in enum_list]
    return enum_list


class NNActivation(Enum):
    RELU = 'relu'
    SOFTMAX = 'softmax'
    TANH = 'tanh'
    SIGMOID = 'sigmoid'
    SWISH = 'swish'
    ELU = 'elu'
    SELU = 'selu'


class NNOptimizers(Enum):
    ADAM = 'Adam'
    SGD = 'SGD'
    RMSPROP = 'RMSprop'
    NADAM = 'Nadam'
    ADADELTA = 'Adadelta'
    ADAGRAD = 'Adagrad'
    ADAMAX = 'Adamax'


class LossFunc(Enum):
    MAE = 'mae'


class Direction(Enum):
    MAXIMIZE = 'maximize'
    MINIMIZE = 'minimize'


class EnsembleMode(Enum):
    NN_SIMPLE_AVG = '1A'
    NN_MEDIAN = '1M'
    NN_WEIGHTED_AVG = '1WA'


class Verbosity(Enum):
    Silent = 0
    Warning = 1
    Info = 2
    Debug = 3
