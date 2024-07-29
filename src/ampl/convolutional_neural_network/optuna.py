from fastai.vision import *
# from fastai import *
import fastai
from fastai.vision.all import *
import torch.nn.modules.activation

from dataclasses import dataclass

import cv2
import os
import joblib
import sklearn
import seaborn as sns
import json
import yaml
import optuna
from statistics import mean
import dill as pickle
import torch
import torch.nn.functional as F

from typing import TYPE_CHECKING

# if TYPE_CHECKING: 
from optuna import pruners

from ampl.enums import *
from ampl.state import State
from ampl.util import Util
from ampl.optuna import OptunaStep
from ampl.constant import Constant as C
from ampl.convolutional_neural_network.util import Util as UtilCNN

import logging

logger = logging.getLogger(__name__)

# start the multiprocessing usin the spawn method
# multiprocessing.set_start_method('spawn', force=True)


@dataclass
class PipelineOptuna(OptunaStep):
  '''
  Optuna Hyper-parameter optimization class
  '''
  state: State
  norm_tfms : bool = True
  apply_tfms : int = 2
  backbone : tuple = (fastai.vision.all.resnet18, fastai.vision.all.resnet34, fastai.vision.all.resnet50, 
                      fastai.vision.all.resnet101, fastai.vision.all.resnet152)
  metrics : tuple = (fastai.metrics.Dice, fastai.metrics.DiceMulti, fastai.metrics.JaccardCoeff)
  act_cls : tuple = (torch.nn.modules.activation.Mish, torch.nn.modules.activation.ReLU, torch.nn.modules.activation.Tanh)
  opt_func : tuple = (fastai.vision.all.ranger, fastai.vision.all.SGD, fastai.vision.all.RMSProp, fastai.vision.all.Adam)
  loss_func : tuple[str] = ('CE', 'F')
  min_lr : float = 0.00001
  max_lr : float = 0.1
  epochs1 : int = 1
  epochs2 : int = 1
  weights : int = 2
  pathData : str = "./data"
  pathModel : str = "./models"
  pathOut : str = "./BasicResNet"
  imgFolder : str = "images"
  labelFolder : str = "targets"
  foldersOpt : int = 2
  foldersSet : str = "Siev_OSM"
  training : bool = True
  output_name: str = "Test"
  filepre: str = "results_all"
  n_trials: int = 3
  directions: tuple[Direction] = (Direction.MAXIMIZE, Direction.MAXIMIZE, Direction.MAXIMIZE)
  timeout: int = 300
  valid_files: tuple = ("valSiev_0.txt", "valSiev_1.txt", "valSiev_2.txt", "valSiev_3.txt", "valSiev_4.txt")
  fn: str = "folder_name"
  foldersSet: tuple[str] = ("Siev_OSM")
  AOI: str = "Siev_OSM"
  allFolders: tuple[str] = (foldersSet) #+['Sievierodonetsk'] #dataset used in testing
  pruner: pruners.MedianPruner = None
  initializer: torch.tensor = None


  def __post_init__(self):
    super().__init__(C.OPTUNA_CNN, self.state, C.OPTUNA_CNN)

    # self.activations = process_enums_list(self.activations)
    # self.optimizers = process_enums_list(self.optimizers)
    self.storage_log = f"{self.state.results_directory}optuna_nn_study_{self.name}_journal.log"

    if self.initializer is None:
      self.initializer = UtilCNN.get_initializer(self.weights, fastai)
    if self.pruner is None:
      optuna = Util.load_optuna()
      self.pruner = optuna.pruners.MedianPruner()

  @dataclass
  class Objective(object):
    """ Declare the variables """
    x_train: np.ndarray
    y_train: np.array
    x_valid: np.ndarray
    y_valid: np.array
    input_shape: List[int]
    loss_weights: torch.tensor
    model_epochs: int
    trial_epochs: int
    trial_min_layers: int = 3
    trial_max_layers: int = 8
    max_neurons: int = 500
    min_lr: float = 1e-3
    max_lr: float = 1e-1
    weights : int = 2
    
    #TODO See if there are more activations, optimizers, and loss functions
    activations: Tuple[CNNActivation] = (CNNActivation.MISH, CNNActivation.RELU, CNNActivation.TANH)
    optimizers: Tuple[CNNOptimizers] = (fastai.vision.all.ranger, fastai.vision.all.Adam, 
                                       fastai.vision.all.RMSProp, fastai.vision.all.SGD)
    
    backbone : Tuple[CNNBackbone] = (fastai.vision.all.resnet18, fastai.vision.all. resnet34, fastai.vision.all.resnet50, 
                                     fastai.vision.all.resnet101, fastai.vision.all.resnet152)
    loss: LossFunc = LossFunc.MAE
    use_kfold: bool = True
       
    def __post_init__(self):
      super().__init__(C.OPTUNA_CNN, self.state, C.CNN)

      # maybe need a different initialize step for pytorch           
      if self.loss_weights is None:
        fa = UtilCNN.load_fastai()
        self.loss_weights = UtilCNN.get_initializer(self.weights, fa)


    def __call__(self, trial, **kwargs):
      """
            Used by the Optuna optimizer to fit and evaluate the model during the early training phase.
            TRIAL_EPOCHS is the number of epochs to run during the initial testing of the model
            to determine if that model should be fully trained in a later step. If it is determined
            that the model should not be fully trained due to lack of early performance, Optuna will prune this trial.

            :param trial: _description_
            :type trial: _type_
            :raises optuna.TrialPruned: _description_
            :return: _description_
            :rtype: _type_
      """
      # TODO May need to put in a way of clearing lcutter from previous pytorch runs here. Awaiting advice from Ashley

      # TODO: check optuna and fastai to make sure they are loading properly
      optuna = Util.load_optuna()
      # TODO May need to load other fastai libraries from Charlotte's code
      fa = UtilCNN.load_fastai()


      # Metrics to be monitored by Optuna
      monitor = "val_loss"

      # TODO maybe optimize the number of layers - What options do we have for manipulating Unet?

      #transformations include normalization?
      norm_tfms = trial.suggest_categorical("norm_tfms", self.boolean_options)

      #3 transformation settings: none, default, more aggressive
      apply_tfms = trial.suggest_int("apply_tfms", self.apply_tfms_min, self.apply_tfms_max, log=False, step=1)
      
      #backbone (generally been looking at resnet18, 50, 101)  
      backbone = trial.suggest_categorical("backbone", self.backbone_options)
      
      # metrics
      metrics = trial.suggest_categorical("metric", self.metric_options)
      
      #activation function
      act_cls = trial.suggest_categorical("activation function", self.activation_function_options)
      
      #optimization function
      opt_func = trial.suggest_categorical("optimization function", self.optimization_function_options)
      
      #loss function (cross-entropy or focal loss) 
      loss_func = trial.suggest_categorical("loss function", self.loss_func_options)
      
      #learning rate (usualy choosen based on fastain's learn.lr_find)
      lr = trial.suggest_float("lr", self.lr_min, self.lr_max, log=True)

      # TODO: i dont know what these 3 variables are for -J
      score_total = 0
      dmg_1_total = 0
      loc_1_total = 0

      # TODO: Check if weights are needed -J
      
      #4 weight settings: no weights, [1,1,1], [1, 1, 10], [1, 15, 35]
      #weights =  trial.suggest_int("weights", weights_min, weights_max, step=1)

      ##########################################################
      ##########################################################
      # TODO: this range section looks very strange to me. I need to compare this with what is happening in bda code
      #  to see if this makes any sense to do

      # iterates throught the validation files -- each file shows the images and targets that will be used for validation in
      # the current iteration of the loop. The idea is to average out any very good or very poor results for a given set of
      # hyperparameters
      for i in range(len(self.parent_object.valFiles)):
        
        # this look iterates through the AOIs (Areas of interest). Each AOI is a different city or a different event. This
        # is intended to allow us to have as much or as little specificity in the range of data that the model is trained on
        for AOI in self.parent_object.foldersSet:
          fn = AOI+str(trial._trial_id)+str(i)
          print(fn)

          # this builds and traings the model. Because it is wrapped in a leaner object (something specific to the fastai library), 
          # we are not building a sequential network by layer like in other parts (e.g. NN) of the code
          self.create_and_train(monitor, norm_tfms, apply_tfms, backbone, metrics, act_cls, opt_func, loss_func, lr, self.parent_object.weights, self.parent_object.AOI, self.parent_object.valFile)

          #get output
          # set output_name in the yaml
          output_name = params["output_name"]
           # TODO: way too much happens in this function. break it down into smaller functions -J
          files, folds = getoutputs(params, self.parent_object.AOI, self.parent_object.valFiles[i], fn, self.parent_object.allFolders, output_name)

          #get output for allFolders datasets
          filepre = params['filepre']
          # TODO: way too much happens in this function. break it down into smaller fucntions
          score, dmg_1, loc_1 = getscore(params, files, folds, fn, AOI, filepre, output_name)

          # TODO: look at exactly what is being returned for these variables and provide a more generic type of output
          score_total = score_total + score
          dmg_1_total = dmg_1_total + dmg_1
          loc_1_total = loc_1_total + loc_1
      
      # the goal is to have the average score for this set of hyperparameters. this means dividing by the total number of loops
      divide_by = len(self.parent_object.valFiles)*len(self.parent_object.foldersSet)
      return score_total/divide_by, dmg_1_total/divide_by, loc_1_total/divide_by
      ##########################################################
      ##########################################################      

  # TODO: this is the optuna section that goes through and looks for things to test. none of this function has been edited for
  #  CNNs at this time. 
  def opt_tune(self, x_train, y_train, x_valid, y_valid, input_shape: List[int]): #, loss_weights: keras.initializers.GlorotUniform):
    """
    Runs the trials within hyperparameter search space looking for the best network configuration
    and records the trial information and optimal neural network configuation from the best trial
    at the conclusion of Optuna search. It also creates a table of the completed trials.

    :param x_train: Training dataset.
    :type x_train: _type_
    :param y_train: Set of labels to all the df in x_train.
    :type y_train: _type_
    :param x_valid: Validation dataset.
    :type x_valid: _type_
    :param y_valid: Set of labels to all the df in x_valid.
    :type y_valid: _type_
    :param input_shape: The shape of the input df provided to the Keras model while training.
    :type input_shape: _type_
    :param loss_weights: Scalar coefficients to weight the loss contributions of different model outputs.
    :type loss_weights: list or dictionary
    :return: Best trial parameter values.
    :rtype: _type_
    """
    optuna = Util.load_optuna()

    storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(self.storage_log))

    ### Create new study with objective and load results into database
    # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html
    study = optuna.create_study(storage=storage,
                                direction=self.direction.value,
                                study_name=self.state.study_name,
                                pruner=self.pruner,
                                load_if_exists=True)

    logger.debug(f'{self.n_jobs =}')
    ### Runs the trials within hyperparameter search space looking for the best network configuration ###################
    with parallel_config(backend='loky', n_jobs=self.n_jobs):
        study.optimize(
            self.Objective(
                x_train,
                y_train,
                x_valid,
                y_valid,
                input_shape=input_shape,
                #loss_weights=loss_weights,
                trial_epochs=self.trial_epochs,
                trial_min_layers=self.trial_min_layers,
                trial_max_layers=self.trial_max_layers,
                max_neurons=self.max_neurons,
                min_lr=self.min_lr,
                max_lr=self.max_lr,
                activations=self.activations,
                optimizers=self.optimizers,
                loss=self.loss),
            n_trials=self.n_trials,
            # n_jobs=self.n_jobs,
            show_progress_bar=True)

    best_params_list = self.process_optuna_study(study)

    return best_params_list
      
#######################################################
#######################################################
# TODO: This is a big nasty fucntion. there are multiple things happening
#  that need to be broken into multiple functions
#create predictions using model, parameters, and specified test dataset
def getoutputs(params, AOI, valFile, fn, foldersSet, output_name):

  #set up model to be able to import trained weights
  outputfolder = str(params['pathOut'])+'/'+fn
  if not os.path.exists(outputfolder):
    os.mkdir(outputfolder)
  #im_folder = str(params['pathData'])+"/"+params['imgFolder']+"/"+AOI
  #targ_folder = str(params['pathData'])+"/"+params['labelFolder']+"/"+AOI

  def get_y(o): 
    return get_image_files(o, recurse = True, folders = foldersSet) #[AOI])
  
  codes = np.loadtxt(Path(params['pathOut'],'codes.txt'), dtype=str)
  
  # print('Get output for: '+AOI)

  bda = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                  get_items = get_y, #get_image_files,
                  get_y = UtilCNN.label_func,
                  splitter=FileSplitter(Path(params['pathOut'], valFile)),
                  batch_tfms= UtilCNN.Normalize2() if params['norm_tfms'] else None
                  #batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)] if apply_tfms else Normalize.from_stats(*imagenet_stats)
              ) 
  dls = bda.dataloaders(Path(params['pathData'],params['imgFolder']), path=params['pathData'], bs=params['bs'], verbose = True, num_workers=0)
  
  # Weights selection (this value is chosen by optuna)
  if params['weights'] == 0: #no weights
    w = None
  elif params['weights'] == 1: #equal weights
    w = torch.tensor([1, 1, 1], device = "cuda").float()
  elif params['weights'] == 2: #weigh damage more
    w = torch.tensor([1, 1, 10], device = "cuda").float()
  else: #approximately inverse to pixel count over city
    w = torch.tensor([1, 50., 250.], device = "cuda").float()

  # Loss function selection (This is chosen by optuna)
  if params['loss_func'] == 'CE':
    ls = CrossEntropyLossFlat(weight = w, axis=1)
  elif params['loss_func'] == 'F':
    ls = FocalLossFlat(weight = w, axis = 1)

  # the unet_learner is the unet model that is implemented using fastai and our selected hyperparameters
  learn = unet_learner(dls, params['backbone'], metrics=params['metrics'], act_cls=params['act_cls'], opt_func=params['opt_func'], loss_func = ls).to_fp16()

  #get neural network weights
  if os.path.isfile(str(params['pathModel'])+'model_2_'+fn+'.pth'):
    print('model 2')
    learn = learn.load(params['pathModel']+'model_2_'+fn)
  else:
    print('model 1')
    learn = learn.load(params['pathModel']+'model_1_'+fn)
  test_dl = dls.valid

  #run prediction
  preds = learn.get_preds(dl=test_dl)

  #save outputs
  outs = outputfolder+'/output_'+output_name
  if(not os.path.isdir(outs)):
    os.mkdir(outs)
  files = []
  folds = []
  for i, pred in enumerate(preds[0]):
    pred_arg = pred.argmax(dim=0).numpy()
    pred_arg = pred_arg.astype(np.uint8)
    im = Image.fromarray(pred_arg)
    im2 = np.array(im)
    plt.imshow(im2)
    val = i
    # print("this is the learn.dls.valid_ds.items string", str(learn.dls.valid_ds.items[i]))
    # print("here's the list of substrings", str(learn.dls.valid_ds.items[i]).split('/'))
    files.append(str(learn.dls.valid_ds.items[i]).split('\\')[-1])
    folds.append(str(learn.dls.valid_ds.items[i]).split('\\')[-2])
    filename = str(learn.dls.valid_ds.items[i]).split('\\')[-1]
    im.save(outs+'/test_damage_'+filename.replace('.png', '_prediction.png').replace('.PNG', '_prediction.png'))
  
  #save localization
  files = os.listdir(outs)
  if(len(files) != len(folds)):
    print("##############################VERY BAD############################")
    print("Dumping files and folders:")
    print("Files: ", files)
    print("Files Length: ", len(files))
    print("Folders: ", folds)
    print("Folders Length: ", len(folds))
  else:
    print("nvm, I don't know where this problem originates")
  for filename in files:
    # print(filename)
    image = cv2.imread(outs+'/'+filename,0)
    image_copy = image.copy() 
    image_copy[image_copy>0] = 1
    cv2.imwrite(outs+'/'+filename.replace('damage', 'localization'),image_copy)
  print('Done with output')
  
  #create human-friendly output
  pred_folder = outputfolder+'/output_'+output_name
  save_folder = outputfolder+'/output_'+output_name+'print'
  if(not os.path.isdir(save_folder)):
    os.mkdir(save_folder)
  for i in range(len(files)):
    filename = files[i]
    if ('damage' in filename) & (filename[-4:] == '.png'):
      
        curr_pred = cv2.imread(pred_folder+'/'+filename, cv2.IMREAD_UNCHANGED)
        curr_pred = curr_pred.astype('float')
        curr_pred[curr_pred == 0] = np.nan
        
        post_file = filename.replace('test_damage_', '').replace('_prediction', '').replace('.png', '.PNG')
        try:
          im_folder = Path(params['pathData'], params['imgFolder'], folds[i])
        except Exception as error:
          print(error)
          print("this is i: ", i)
          print("this is the length of the folders: ", len(folds))
          print("this is the length of the files: ", len(files))

        if not os.path.isfile(Path(im_folder, post_file)):
          post_file = post_file.replace('.PNG', '.png')
        # print(Path(im_folder, post_file))
        curr_post= cv2.imread(str(Path(im_folder, post_file)), cv2.IMREAD_UNCHANGED)
        curr_post = cv2.cvtColor(curr_post, cv2.COLOR_BGR2RGB) 
        
        try:
          targ_folder = str(Path(params['pathData'], params['labelFolder'], folds[i]))
        except Exception as error:
          print(error)
          print("this is i: ", i)
          print("this is the length of the folders: ", len(folds))
          print("this is the length of the files: ", len(files))
        targ_file = post_file.replace('.', 'd.')
        curr_targ = cv2.imread(targ_folder+'/'+targ_file, cv2.IMREAD_UNCHANGED)
        curr_targ = curr_targ.astype('float')
        curr_targ[curr_targ == 0] = np.nan        

        cmap = matplotlib.colors.ListedColormap(["white","yellow","orange","red"])        
        f, ax = plt.subplots(1,2, figsize = (10,10))
        ax[0].imshow(curr_post)
        ax[0].imshow(curr_targ, alpha = 0.7, cmap=cmap)
        ax[0].axis('off')
        ax[0].title.set_text('Ground Truth')
        ax[1].imshow(curr_post)
        ax[1].imshow(curr_pred, alpha = 0.7, cmap=cmap)
        ax[1].axis('off')
        ax[1].title.set_text('Prediction')
        plt.savefig(save_folder+'/'+filename.replace('test_damage', 'viz_pred').replace('_prediction', ''))
        
  print('Done with printer friendly version')
  return files, folds 
  #########################################################
  #########################################################
  ### TODO: another massive fucntion that needs to be broken into pieces
  #get score
def getscore(params, files, folds, fn, AOI, filepre, output_name):
  outputfolder = Path(params['pathOut'], fn)
  probFolder = str(Path(outputfolder, 'output_'+output_name))
  # print(probFolder)
  
  # print(str(probFolder).split('/'))
  out_folder = "/".join(str(probFolder).split('/')[:-1])
  columns = ['pred', 'gt']
  dfs = []
  for i in range(len(files)):
    filename = files[i]
    if ('damage' in filename):
      df_curr = pd.DataFrame(columns = columns)
      curr_pred = cv2.imread(probFolder+'/'+filename, cv2.IMREAD_UNCHANGED)
      curr_pred_flat = curr_pred.flatten()
      df_curr['pred'] = curr_pred_flat
        
        
      #get ground truth
      gt_file = filename.replace('test_damage_', '').replace('_prediction', 'd').replace('.png', '.PNG')
      targ_folder = str(Path(params['pathData'], params['labelFolder'], folds[i]))
      f = targ_folder+'/'+gt_file
      if not os.path.isfile(f):
        f = f.replace('.PNG', '.png')
      curr_gt = cv2.imread(f, cv2.IMREAD_UNCHANGED)
      df_curr['gt'] = curr_gt.flatten()
      dfs.append(df_curr)
  df = dfs[0]
  for i in range(1, len(dfs)):
    df = df.append(dfs[i])
  df = df.reset_index()
  df = df.drop('index', axis = 1)

  df = df.loc[df['gt'] != 3].copy()
  df['pred_loc'] = df['pred']
  df.loc[((df['pred'] >0) & (df['pred'] <3)) , 'pred_loc'] = 1
  df['gt_loc'] = df['gt']
  df.loc[(df['gt_loc'] >0 ) , 'gt_loc'] = 1

  #localization confustion matrix
  gt_labels = df['gt_loc'].to_numpy()
  res_labels = df['pred_loc'].to_numpy()
  print(sklearn.metrics.confusion_matrix(gt_labels, res_labels))
  options = ['no building', 'building']
  plt.close() 
  ax = sns.heatmap(sklearn.metrics.confusion_matrix(gt_labels, res_labels), annot=True,fmt=".0f", xticklabels = options, yticklabels = options,  cmap='Blues')
  ax.set(xlabel="Predicted", ylabel="Actual", title = "Confusion Matrix")
  ax.figure.tight_layout()
  plt.savefig(Path(out_folder, filepre+'_cm_loc.png'))
  plt.close() 

  #localization f1
  f1s = [0,0]
  for i in range(2):
    pred_curr = res_labels == i
    pred_curr = pred_curr.astype(int)
    gt_curr = gt_labels == i
    gt_curr = gt_curr.astype(int)
    curr_f1 = sklearn.metrics.f1_score(gt_curr, pred_curr)
    f1s[i] = curr_f1
    print('f1 ' + options[i] + ' :' + str(curr_f1))
  loc_f1 = f1s[1]

  #damage in buildings
  df_damage = df.copy()
  df_damage = df_damage[df_damage['gt_loc'] == 1]
  #damage in buildings confusion amtrix
  gt_labels = df_damage['gt'].to_numpy()
  res_labels = df_damage['pred'].to_numpy()
  print(sklearn.metrics.confusion_matrix(gt_labels, res_labels))
  options = ['no building', 'no damage', 'damage']
  ax = sns.heatmap(sklearn.metrics.confusion_matrix(gt_labels, res_labels)[1:,1:], annot=True,fmt=".0f", xticklabels = options[1:], yticklabels = options[1:],  cmap='Blues')
  ax.set(xlabel="Predicted", ylabel="Actual", title = "Confusion Matrix")
  ax.figure.tight_layout()
  plt.savefig(Path(out_folder, filepre+'_cm_dmg.png'))
  plt.close() 
  #damage in buildings f1
  f1s = [0,0,0]
  for i in range(3):
    pred_curr = res_labels == i
    pred_curr = pred_curr.astype(int)
    gt_curr = gt_labels == i
    gt_curr = gt_curr.astype(int)
    curr_f1 = sklearn.metrics.f1_score(gt_curr, pred_curr)
    f1s[i] = curr_f1
    print('f1 ' + options[i] + ' :' + str(curr_f1))
  dmg_f1 = scipy.stats.hmean(f1s[1:3])
  score = dmg_f1*.7 + loc_f1*.3
  print('mean f1 :'+str(dmg_f1))
  print('Score: '+str(score))
  
  #damage raw confusion matrix
  gt_labels = df['gt'].to_numpy()
  res_labels = df['pred'].to_numpy()
  print(sklearn.metrics.confusion_matrix(gt_labels, res_labels))
  ax = sns.heatmap(sklearn.metrics.confusion_matrix(gt_labels, res_labels), annot=True,fmt=".0f", xticklabels = options, yticklabels = options,  cmap='Blues')
  ax.set(xlabel="Predicted", ylabel="Actual", title = "Confusion Matrix")
  ax.figure.tight_layout()
  plt.savefig(Path(out_folder, filepre+'_cm_rawdmg.png'))

  f1s_raw = [0,0,0]
  for i in range(3):
    pred_curr = res_labels == i
    pred_curr = pred_curr.astype(int)
    gt_curr = gt_labels == i
    gt_curr = gt_curr.astype(int)
    curr_f1 = sklearn.metrics.f1_score(gt_curr, pred_curr)
    f1s_raw[i] = curr_f1
    print('f1 ' + options[i] + ' :' + str(curr_f1))
    

  #json
  j = { "score": score,
    "damage_f1": dmg_f1,
    "localization_f1": loc_f1,
    "damage_f1_no_damage": f1s[1],
    "damage_f1_damage": f1s[2],
    "raw_f1_noBuilding": f1s_raw[0],
    "raw_f1_noDamage": f1s_raw[1],
    "raw_f1_Damage": f1s_raw[2],
    }
  json_object = json.dumps(j)
  # TODO change this to not have filepre in it
  # just use the full filename
  with open(Path(out_folder, filepre+"_results.json"), "w") as outfile:
    outfile.write(json_object)
  
  return score, dmg_f1, loc_f1
  
  
  #########################################################
  #########################################################

 