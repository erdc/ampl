from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    import tensorflow.keras as keras


from ampl.constant import Constant as C
from ampl.enums import NNOptimizers, LossFunc
from ampl.state import State
from ampl.pipelinestep import PipelineStep
from ampl.util import Util
from ampl.neural_network.util import Util as UtilNN

import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineModelBuild(PipelineStep):
    """
        Create the ensemble of good models to perform predictions with. This code requires models to be in
        the database to perform different ensemble methods.

        :type patience: int - Patience value to pass to NN model Early Stopping
        :type loss: LossFunc - Type of LossFunction to use in NN Model
    """

    state: State
    loss: LossFunc = LossFunc.MAE
    metrics: List[str] = field(default_factory=lambda: ['mse'])
    epochs: int = 1000
    patience: int = 50
    models: List[keras.Sequential] = field(default_factory=list, init=False)
    early_stopping: keras.callbacks.EarlyStopping = None
    reduce_lr: keras.callbacks.ReduceLROnPlateau = None
    verbose: int = 0

    def __post_init__(self):
        super().__init__(C.BUILD_NN, self.state, C.OPTUNA_NN)
        tf = UtilNN.load_tensorflow()
        self.loss = self.loss.value
        if self.early_stopping is None:
            self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, verbose=self.verbose,
                                                mode='min', min_delta=0.00001)
        if self.reduce_lr is None:
            self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0000001,
                                               verbose=self.verbose, mode='min')

    def get_model(self, top_trial, X_train, X_val, y_train, y_val):
        tf = UtilNN.load_tensorflow()

        # Build Neural Network from Optuna results
        input_shape = (X_train.shape[1],)
        top_trial = top_trial.dropna()
        optimizer = UtilNN.get_keras_optimizer(top_trial[C.LEARNING_RATE],
                                        NNOptimizers(top_trial[C.OPTIMIZER]))
        activation = UtilNN.get_keras_activation(top_trial[C.ACTIVATION])

        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=input_shape))
        model.add(tf.keras.layers.Dense(int(top_trial[C.N_UNITS_INPUT_LAYER]), activation=activation))

        for i in range(top_trial[C.N_LAYERS]):
            model.add(tf.keras.layers.Dense(int(top_trial[f'{C.N_UNITS_LAYER}{i:02}']), activation=activation))

        model.add(tf.keras.layers.Dense(1))

        model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)
        t_start = time.time()
        history = model.fit(X_train, y_train, epochs=self.epochs, validation_data=(X_val, y_val),
                            callbacks=[self.reduce_lr, self.early_stopping])
        t_end = time.time()
        return model, t_end - t_start

    def run(self, random_state: int = 0, model_ext: str = C.MODEL_EXT_NN):
        logging.debug('Running NN Build step')
        Util.check_and_create_directory(self.state.results_directory)
        Util.check_and_create_directory(self.state.saved_models_directory)

        X_train, X_val, X_test, y_train, y_val, y_test = self.state.data.train_val_test_split(random_state=random_state)

        self.journal[C.TRAINING_POINTS] = X_train.shape[0]
        self.journal[C.TESTING_POINTS] = X_test.shape[0]
        self.journal[C.VALIDATION_POINTS] = X_val.shape[0]

        input_shape = (X_train.shape[1],)
        logging.debug(f'{input_shape = }')

        best_trial_df, top_trials_df = self.load_trials_df()

        self.journal[C.EPOCHS_SET] = self.epochs
        self.journal[C.N_RUNS] = self.state.num_models

        for j in range(self.state.num_models):
            model, t_run = self.get_model(top_trials_df.iloc[j], X_train, X_test, y_train, y_test)
            history = model.history
            self.models.append(model)

            saved_model_file = self.state.get_model_path(j, model_ext=model_ext)
            file_base = self.state.get_model_base_path(j)
            saved_history_file = file_base + C.HISTORY_EXT

            model.save(saved_model_file)

            with open(saved_history_file, 'wb') as history_file:
                pickle.dump(history.history, history_file)

            self.journal[f'run_{j}'] = {
                C.RUN_TIME: t_run,
                C.RUN_EPOCHS: len(history.history['loss'])
            }
