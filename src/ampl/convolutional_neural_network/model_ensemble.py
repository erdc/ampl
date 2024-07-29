from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

import fastai.callback

# removed TYPE_CHECKING
from optuna import pruners
import fastai
import fastai.vision
from fastai.vision.all import *

from ampl.constant import Constant as C
# TODO: look and see what optimizers the CNN uses from BDA
from ampl.enums import CNNOptimizers, LossFunc
from ampl.state import State
from ampl.pipelinestep import PipelineStep
from ampl.util import Util
from ampl.convolutional_neural_network.util import Util as UtilCNN


import os
import numpy as np

import logging

logger = logging.getLogger(__name__)


    
def label_func(o):
    return  Path(str(o.parent).replace('images', 'targets'))/f'{o.stem}d{o.suffix}'#how to get label file name from image file name
    
@dataclass 
class PipelineModelEnsemble(PipelineStep):
    '''
    Create the ensemble of good models to perform predictions with. This code requires models to be in
        the database to perform different ensemble methods.

        :type patience: int - Patience value to pass to CNN model Early Stopping
        :type loss: LossFunc - Type of LossFunction to use in NN Model
    '''

    state: State_CNN
    loss: LossFunc = LossFunc.MAE
    metrics: List[str] = field(default_factory=lambda: ['mse'])
    epochs: int = 1000
    patience: int = 50
    models: List[fastai.vision.Learner] = field(default_factory=list, init = False)
    early_stopping: fastai.callback = None
    params: dict = None
    reduce_lr: fastai.callback = None

    def __post_init__(self):
        super().__init__(C.BUILD_CNN, self.state, C.OPTUNA_CNN)
        fa = UtilCNN.load_fastai()
        self.loss = self.loss.value
        if self.early_stopping is None:
            self.early_stopping = EarlyStoppingCallback(monitor='valid_loss', comp=None, min_delta=0.0,
                        patience=1, reset_on_fit=True)
        if self.reduce_lr is None:
            self.reduce_lr = ReduceLROnPlateau(monitor='valid_loss', comp=None, min_delta=0.0, patience=1,
                        factor=10., min_lr=0, reset_on_fit=True)
    
    def train_model(self, params, AOI, allFolders, valFile, fn, top_trial):
        top_trial = top_trial.dropna()
        # TODO check whether training bool is set here
        # it's unclear if anything important happens with the inputs if training is set to false

        os.getcwd()
        codes = np.loadtxt('codes.txt', dtype=str) #get label names
        fnames = get_image_files(Path(params['pathData'],params['imgFolder']))
        #label_func = lambda o: params['pathData']/params['labelFolder']/str(o.parent).split('\\')[-1]/f'{o.stem}d{o.suffix}' #how to get label file name from image file name

        #3 transformation settings: none, default, more aggressive
        #two more optuna-able parameters here- apply_tfms and norm_tfms
        if top_trial['apply_tfms'] > 0:
            if top_trial['apply_tfms'] == 1:
                tfs = [*aug_transforms(), Normalize2() if params['norm_tfms'] else None]
            else:
                tfs = [*aug_transforms(flip_vert= True, max_rotate = 30, max_zoom = 1.2, min_zoom = 0.8, max_lighting = .5), 
            Contrast(max_lighting = 0.5, p = 0.5), Saturation(max_lighting = 0.5, p = 0.5), 
            Hue(max_hue = 0.5, p = 0.5), Normalize2() if params['norm_tfms'] else None]
        else:
            if top_trial['norm_tfms']:
                tfs = Normalize2()
            else:
                tfs = None
        #3 folder settings: train on current folder/dataset, train on alxl but current folder/dataset in 'allFolders', or train on datasets in 'allFolders'
        if params['foldersOpt'] == 0:
            folders = [AOI]
        elif params['foldersOpt'] == 1:
            folders = allFolders.copy()
            folders.remove(AOI)
        elif params['foldersOpt'] == 2:
            folders = allFolders
        else:
            print('no valid folders')
        #get hold-out set/test image file
        valid = Path(params['pathOut'], valFile)

        def get_y(o): return get_image_files(o, recurse = True, folders = folders)

        #create dataset
        bda = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                        get_items = get_y,
                        get_y = label_func,
                        splitter=FileSplitter(valid),
                        batch_tfms= tfs
                    ) 
        dls = bda.dataloaders(Path(params['pathData'],params['imgFolder']), path=params['pathData'], bs=params['bs'], verbose = True, num_workers=0)

        # shape = dls.dataset.data.shape  
        # print(shape)

        dls.show_batch()
        plt.savefig('batch_'+fn+'.png')

        #setup neural net
        #4 weight settings: no weights, [1,1,1], [1, 1, 10], [1, 11, 35]
        #optuna variable
        if top_trial['weights'] == 0: #no weights
            w = None
        elif top_trial['weights'] == 1: #equal weights
            w = torch.tensor([1, 1, 1], device = "cuda").float()
        elif top_trial['weights'] == 2: #weigh damage more
            w = torch.tensor([1, 1, 10], device = "cuda").float()
        else: #approximately inverse to pixel count over Sievierodonetsk
            w = torch.tensor([1, 15., 30.], device = "cuda").float()
        print(w)
        
        #loss function (cross-entropy or focal loss)
        #optunat variable
        if top_trial['loss_func'] == 'CE':
            ls = CrossEntropyLossFlat(weight = w, axis=1)
        else:
            ls = FocalLossFlat(weight = w, axis = 1)
        # unet_learner is a model using the backbone as a basis. it downsamples and convolves the data down, and then upsamples back up to have an output 
        # that is the size of the input image.
        learn = unet_learner(dls, params['backbone'], metrics=params['metrics'], act_cls=params['act_cls'], opt_func=params['opt_func'], loss_func = ls).to_fp16()
        print(learn.summary())

        t_start = time.time()
        #first train last layer
        # TODO should allow user to choose fit function - fit_flat_cos should be the default option
        learn.fit_flat_cos(params['optuna_epochs'],slice(params['lr']), cbs=[GradientAccumulation(n_acc=8), EarlyStoppingCallback (monitor='valid_loss', comp=None, min_delta=0.0,
                            patience=1, reset_on_fit=True)]) #optuna variable
        t_end = time.time()

        # TODO make sure to save this to results folder
        learn.show_results(max_n=6, figsize=(7,8))
        plt.savefig('model_1_'+fn+'.png')
        learn.save(params['pathModel']+'model_1_'+fn)
        #unfreeze and train full network
        lrs = slice(params['lr']/400, params['lr']/4)
        learn.unfreeze()

        return learn, t_end - t_start
    

    def get_learner(self, top_trial, AOI, allFolders, valFile):
        fa = UtilCNN.load_fastai()

        # build unet_learner
        top_trial = top_trial.dropna()
        optimizer = UtilCNN.get_fastai_optimizer()
        activation = UtilCNN.get_fastai_activation()

        fn = "model_build_{AOI}_{val_file}"
   
        # We are no longer doing kfold, so we will only be using a single valFile
        learn, elapsed_time = self.train_model(params, AOI, allFolders, valFile, fn, fa)

        return learn, elapsed_time
    
    def run(self, random_state: int = 0, model_ext: str = C.MODEL_EXT_CNN):
        logging.debug('Running CNN Build Step')
        Util.check_and_create_directory(self.state.results_directory)
        Util.check_and_create_directory(self.state.saved_models_directory)

        #TODO move data splitting out here from the model train function

        #TODO get the valFile, AOI, and allFolders values from the parameter file.
        AOI = ""
        allFolders = ""
        valFile = ""

        best_trial_df, top_trials_df = self.load_trials_df()

        self.journal[C.EPOCHS_SET] = self.epochs
        self.journal[C.N_RUNS] = self.state.num_models

        for j in range(self.state.num_models):
            learner, t_run = self.get_learner(top_trials_df.iloc[j], AOI, allFolders, valFile)

            file_base = self.state.get_model_base_path(j)
            saved_history_file = file_base + '.pickle'

            learner.save(saved_history_file, with_opt=True, pickle_protocol=2)

            self.journal[f'run_{j}'] = {
                C.RUN_TIME: t_run,
                # TODO find a way to record the number of epochs
                # C.RUN_EPOCHS: len(history.history['loss'])
            }