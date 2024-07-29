from ampl.enums import CNNActivation, CNNOptimizers

from fastai.vision import *
from fastai import *
from fastai.vision.all import *
import torch.nn.modules.activation as act

    
class Util(object):
    @staticmethod
    def load_fastai():
        """
        Lazy loading fastai
        :return: fastai module
        """
        import os
        import sys

        if sys.platform == 'darwin':
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        os.environ['LANG'] = "en_US.UTF-8"
        os.environ['LANGUAGE'] = "en_US.UTF-8"
        os.environ['LC_ALL'] = "en_US.UTF-8"

        import fastai.vision.all
        return fastai.vision.all
    
    @staticmethod
    def get_initializer(weights, fastai):

        if weights == 0: #no weights
            w = None
        elif weights == 1: #equal weights
            w = torch.tensor([1, 1, 1], device = "cuda").float()
        elif weights == 2: #weigh damage more
            w = torch.tensor([1, 1, 10], device = "cuda").float()
        else: #approximately inverse to pixel count over Sievierodonetsk
            w = torch.tensor([1, 15., 30.], device = "cuda").float()
        return w

    @staticmethod
    def get_fastai_activation(activation: CNNActivation = CNNActivation.RELU):
        fa = Util.load_fastai()

        if isinstance(activation, str):
            activation = CNNActivation(activation)

        activations = {
            CNNActivation.MISH: fa.nn.modules.activation.Mish,
            CNNActivation.RELU: fa.nn.modules.activation.ReLU,
            CNNActivation.TANH: fa.nn.modules.activation.Tanh
        }
        return activations.get(CNNActivation(activation), activations[CNNActivation.RELU])

    @staticmethod
    def get_fastai_optimizer(my_optimizer: CNNOptimizers = CNNOptimizers.ADAM):
        '''
            Get the optimizer used in the best model trial. From time to time new optimizers will be released
            by fastai. If you would like to use an optimizer then add it to the opts object. On error the default Optimizer
            is Adam

            :param my_optimizer:
            :return: fastai optimizer
            :rtype: _type_
        '''
        if isinstance(my_optimizer, str):
            my_optimizer = CNNOptimizer(my_optimizer)
        
        fa = Util.load_fastai()

        optimizers = {
            CNNOptimizers.RANGER: fa.ranger,
            CNNOptimizers.ADAM : fa.Adam,
            CNNOptimizers.RMSPROP : fa.RMSProp,
            CNNOptimizers.SGD : fa.SGD
        }
        return optimizers.get(CNNOptimizers(my_optimizer), optimizers[CNNOptimizers.ADAM])
    
    # J- utility - normalize function modified from fastai
    #basic Normalize function from fastai seems to use a random batch to set mean and standard dev
    #(set mean to 0 and standard dev to 1)
    #https://github.com/fastai/fastai/blob/master/fastai/data/transforms.py#L364
    class Normalize2(DisplayedTransform):
        "Normalize/denorm batch of `TensorImage`"
        # L is a shorthand function provided by fastai that was inspried by numpy lists
        parameters,order = L('mean', 'std'),99
        def __init__(self, mean=None, std=None, axes=(0,2,3)): store_attr()

        @classmethod
        def from_stats(cls, mean, std, dim=1, ndim=4, cuda=True): return cls(*broadcast_vec(dim, ndim, mean, std, cuda=cuda))

        def setups(self, dl:DataLoader):
            if self.mean is None or self.std is None:
                x,*_ = dl.one_batch()
                self.mean,self.std = x.mean(self.axes, keepdim=True),x.std(self.axes, keepdim=True)+1e-7 #this is the line I changed

        def encodes(self, x:TensorImage): 
            return (x-x.mean(self.axes, keepdim=True)) / (x.std(self.axes, keepdim=True)+1e-7)
    
    # get label file name from image file name
    @staticmethod
    def label_func(o):
        return  Path(str(o.parent).replace('images', 'targets'))/f'{o.stem}d{o.suffix}'
    ################################