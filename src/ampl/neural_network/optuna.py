from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple, List, Any

import numpy as np
from joblib import parallel_config

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tensorflow.keras as keras
    from optuna import pruners

from sklearn.model_selection import train_test_split

from ampl.enums import *
from ampl.state import State
from ampl.util import Util
from ampl.neural_network.util import Util as UtilNN
from ampl.constant import Constant as C

from ampl.optuna import OptunaStep

import json

import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineOptuna(OptunaStep):
    """ Optuna Hyper-parameter optimization class """
    state: State
    """ State class """

    n_trials: int = 500
    """ Number of trials to use in Optuna study"""

    trial_epochs: int = 20
    """ Number of epochs to use in each Optuna trial"""

    direction: Direction = Direction.MAXIMIZE
    """ Optuna study direction"""

    pruner: pruners.MedianPruner = None
    """ Optuna Pruner to use"""

    trial_min_layers: int = 3
    """ Minimum numbers of layers to use in study """

    trial_max_layers: int = 8
    """ Maximum numbers of layers to use in study """

    max_neurons: int = 500
    """ Maximum neurons each layer to use in study"""

    min_lr: float = 1e-3
    """ Minimum learning rate to use in Optuna study"""

    max_lr: float = 1e-1
    """ Maximum learning rate to use in Optuna study"""

    activations: Tuple[NNActivation] = (
        NNActivation.RELU, NNActivation.SOFTMAX, NNActivation.TANH, NNActivation.SIGMOID,
        NNActivation.SWISH, NNActivation.ELU, NNActivation.SELU)
    """ Tensorflow Activations to use in Optuna Study"""

    optimizers: Tuple[NNOptimizers] = (
        NNOptimizers.ADAM, NNOptimizers.SGD, NNOptimizers.RMSPROP, NNOptimizers.NADAM,
        NNOptimizers.ADADELTA, NNOptimizers.ADAGRAD, NNOptimizers.ADAMAX)
    """ Tensorflow Optimizers to use in Optuna Study"""
    loss: LossFunc = LossFunc.MAE
    """ Tensorflow Loss functions to use in Optuna Study"""

    initializer: keras.initializers.GlorotUniform = None
    """ Tensorflow Initializer to use in Optuna Study, default set to tf.keras.initializers.GlorotUniform()"""

    n_jobs: int = -1
    """ Number of parallel processes to use in Optuna study, set -1 to use what's available, default set to -1"""

    def __post_init__(self):
        super().__init__(C.OPTUNA_NN, self.state, C.NN)

        self.activations = process_enums_list(self.activations)
        self.optimizers = process_enums_list(self.optimizers)
        self.storage_log = f"{self.state.results_directory}optuna_nn_study_{self.name}_journal.log"

        if self.initializer is None:
            tf = UtilNN.load_tensorflow()
            self.initializer = tf.keras.initializers.GlorotUniform()
        if self.pruner is None:
            optuna = Util.load_optuna()
            self.pruner = optuna.pruners.MedianPruner()

    @dataclass
    class Objective(object):
        """
            Accepts model information for initialization.

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
            :type loss_weights: _type_
        """
        x_train: np.ndarray
        y_train: np.array
        x_valid: np.ndarray
        y_valid: np.array
        input_shape: List[int]
        loss_weights: keras.initializers.GlorotUniform
        trial_epochs: int
        trial_min_layers: int = 3
        trial_max_layers: int = 8
        max_neurons: int = 500
        min_lr: float = 1e-3
        max_lr: float = 1e-1
        activations: Tuple[NNActivation] = (
        NNActivation.RELU, NNActivation.SOFTMAX, NNActivation.TANH, NNActivation.SIGMOID,
        NNActivation.SWISH, NNActivation.ELU, NNActivation.SELU)
        optimizers: Tuple[NNOptimizers] = (
        NNOptimizers.ADAM, NNOptimizers.SGD, NNOptimizers.RMSPROP, NNOptimizers.NADAM,
        NNOptimizers.ADADELTA, NNOptimizers.ADAGRAD, NNOptimizers.ADAMAX)
        loss: LossFunc = LossFunc.MAE

        def __post_init__(self):
            self.number_of_features = self.x_train.shape[1]
            if self.loss_weights is None:
                tf = UtilNN.load_tensorflow()
                self.loss_weights = tf.keras.initializers.GlorotUniform()

        # store training and validation datasets

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
            optuna = Util.load_optuna()
            tf = UtilNN.load_tensorflow()

            # Metrics to be monitored by Optuna.
            monitor = C.OPTUNA_MONITOR

            ### optimize number of layers
            ### Suggest an int for the n_layers parameter.
            n_layers = trial.suggest_int("n_layers", self.trial_min_layers, self.trial_max_layers)

            ### Sequential model is a plan stack of layers which each layer has exactly one input tensor and one output tensor
            model = tf.keras.Sequential()

            # print(f"    Number of features: {n_features}")
            max_size = trial.suggest_int(C.N_UNITS_INPUT_LAYER, self.number_of_features,
                                         self.max_neurons)  # number of neurons in input layer

            ### example of how to vary the activation function
            activation = trial.suggest_categorical(C.ACTIVATION, self.activations)

            ### add initial dense layer with set parameters
            model.add(tf.keras.Input(self.input_shape))
            model.add(tf.keras.layers.Dense(max_size, name='input_layer_', activation=activation))

            # create additional layers chosen by optuna
            for i in range(n_layers):
                model.add(tf.keras.layers.Dense(trial.suggest_int(f"{C.N_UNITS_LAYER}{i:02}", self.number_of_features, max_size),
                                             name=f'hidden_{i:02}', activation=activation))

            model.add(tf.keras.layers.Dense(1, name='output_layer'))

            learning_rate = trial.suggest_float(C.LEARNING_RATE, self.min_lr, self.max_lr)
            optimizer_str = trial.suggest_categorical(C.OPTIMIZER, self.optimizers)

            # Get Keras Optimizer object with Optuna suggested learning rate
            optimizer = UtilNN.get_keras_optimizer(learning_rate, optimizer_str)

            model.compile(loss=self.loss.value, optimizer=optimizer, **kwargs)

            # Create callbacks for early stopping and pruning.
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor=C.EARLY_STOPPING_MONITOR, patience=C.EARLY_STOPPING_PATIENCE),
                optuna.integration.TFKerasPruningCallback(trial, monitor),
            ]
            history = model.fit(self.x_train, self.y_train, validation_data=(self.x_valid, self.y_valid),
                                epochs=self.trial_epochs, callbacks=callbacks)

            return history.history[monitor][-1]

    # ###############################################################################

    def opt_tune(self, x_train, y_train, x_valid, y_valid, input_shape: List[int], loss_weights: keras.initializers.GlorotUniform):
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
                    loss_weights=loss_weights,
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

    def run(self, random_state: int = 0):
        """
        Starts running the code, importing all modules that the program needs.
        """
        # self.load_data()
        Util.check_and_create_directory(self.state.results_directory)

        logger.debug(f'training fraction = {self.data.train_size}')
        logger.debug(f'test_size = { (self.data.test_size + self.data.val_size)}')

        X_train, _, X_test, y_train, _, y_test = self.data.train_val_test_split()

        self.journal[C.TRAINING_POINTS] = X_train.shape[0]
        self.journal[C.TESTING_POINTS] = X_test.shape[0]
        self.journal[C.TRAIN_FRAC] = len(X_train)
        self.journal[C.TEST_FRAC] = len(X_test)

        # input shape and weights
        model_input_shape = (X_train.shape[1],)
        logging.debug(f'{model_input_shape = }')
        initializer = self.initializer
        weights = initializer(shape=(X_train.shape[0],))

        # initial time
        t0 = time.time()
        # calling the function that performs the hyperparameter optimization to get the best network configuration
        best_params = self.opt_tune(X_train, y_train, X_test, y_test, model_input_shape, weights)

        t_tune = time.time() - t0

        self.journal[C.RUN_TIME] = t_tune
