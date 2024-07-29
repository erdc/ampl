from enum import Enum

## CNN inports for quick reference to vision functions. May remove later
import fastai.vision.all 
import torch
from torch import nn, optim


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

class CNNOptimizers(Enum):   
    RANGER = fastai.vision.all.ranger
    SGD = fastai.vision.all.SGD
    RMSPROP = fastai.vision.all.RMSProp
    ADAM = fastai.vision.all.Adam
    
class CNNBackbone(Enum):
    RESNET18 = fastai.vision.all.resnet18
    RESNET34 = fastai.vision.all.resnet34 
    RESNET50 = fastai.vision.all.resnet50 
    RESNET101 = fastai.vision.all.resnet101 
    RESNET152 = fastai.vision.all.resnet152

class CNNActivation(Enum):
    RELU = torch.nn.modules.activation.ReLU
    MISH = torch.nn.modules.activation.Mish
    TANH = torch.nn.modules.activation.Tanh

class CNNMetrics(Enum):
    DICE = fastai.metrics.Dice
    DICEMULTII = fastai.metrics.DiceMulti
    JACCARDCOEFF = fastai.metrics.JaccardCoeff
    

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
