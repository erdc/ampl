from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from joblib import parallel_config

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna

from ampl.enums import Direction
from ampl.state import State
from ampl.util import Util
from ampl.optuna import OptunaStep
from ampl.constant import Constant as C

import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineOptuna(OptunaStep):
    """
    """
    state: State
    n_trials: int = 500
    loss: str = "reg:squarederror"
    n_estimator: int = 10000
    eval_metric: str = 'rmse'
    early_stopping_rounds: int = 200
    observation_key: str = 'validation_0-rmse'
    direction: Direction = Direction.MAXIMIZE
    pruner: optuna.pruners.MedianPruner = None
    sampler: optuna.samplers.TPESampler = None
    multivariate: bool = False
    n_jobs: int = -1

    def __post_init__(self):
        super().__init__(C.OPTUNA_DT, self.state, C.OPTUNA_DT)
        if self.sampler is None:
            optuna = Util.load_optuna()
            self.sampler = optuna.samplers.TPESampler(seed=0, multivariate=self.multivariate)

        self.storage_log = f"{self.state.results_directory}optuna_dt_study__{self.name}_journal.log"

        if self.pruner is None:
            optuna = Util.load_optuna()
            self.pruner = optuna.pruners.MedianPruner()

    @dataclass
    class Objective:
        x_train: np.ndarray
        y_train: np.array
        x_valid: np.ndarray
        y_valid: np.array
        early_stopping_rounds: int = 200
        train_frac: float = 0.8
        loss: str = "reg:squarederror"
        eval_metric: str = 'rmse'
        observation_key: str = 'validation_0-rmse'
        n_estimator: int = 10000

        def __call__(self, trial, **kwargs):
            optuna = Util.load_optuna()

            """
            Used by the Optuna optimizer to fit and evaluate the model during the early training phase.

            """
            params = {
                "verbosity": 0,  # 0 (silent) - 3 (debug)
                "objective": self.loss,
                "n_estimators": trial.suggest_int('n_estimators', 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.6, log=True),
                "subsample": trial.suggest_float("subsample", 0.4, 0.8, log=True),
                "alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True),
                "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
                "gamma": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
                "min_child_weight": trial.suggest_float("min_child_weight", 10, 1000, log=True),
                "seed": 24
                # TODO Utilize parallel processing features in
            }

            model = XGBRegressor(**params)

            model.fit(
                self.x_train,
                self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_valid, self.y_valid)],
                eval_metric=self.eval_metric,
                verbose=0,
                callbacks=[optuna.integration.XGBoostPruningCallback(trial, self.observation_key)],
                early_stopping_rounds=self.early_stopping_rounds)

            y_pred_valid = model.predict(self.x_valid)

            return np.sqrt(mean_squared_error(self.y_valid, y_pred_valid))

    def opt_tune(self, x_train, y_train, x_valid, y_valid, ):
        optuna = Util.load_optuna()
        ########################### Create a new study with Objective and load results into database #######################
        storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(self.storage_log))

        SAMPLER = optuna.samplers.TPESampler(seed=0, multivariate=self.multivariate)
        study = optuna.create_study(direction=self.direction.value,
                                    study_name=self.state.study_name,
                                    pruner=self.pruner,
                                    sampler=self.sampler,
                                    storage=storage,
                                    load_if_exists=True)

        logger.debug(f'{self.n_jobs =}')
        ### Runs the trials within hyperparameter search space looking for the best network configuration ###################
        with parallel_config(backend='loky', n_jobs=self.n_jobs):
            study.optimize(self.Objective(x_train,
                                          y_train,
                                          x_valid,
                                          y_valid,
                                          early_stopping_rounds=self.early_stopping_rounds,
                                          train_frac=self.data.train_size,
                                          loss=self.loss,
                                          eval_metric=self.eval_metric,
                                          observation_key=self.observation_key,
                                          n_estimator=self.n_estimator),
                           n_trials=self.n_trials,
                           # n_jobs=self.n_jobs,
                           show_progress_bar=True)

        best_params_list = self.process_optuna_study(study)

        return best_params_list

    @property
    def optuna_results_file(self):
        """
        Sets the path to the file for recording the Optuna results.
        This file contains the optimal neural network configuration.
        """
        if self._optuna_results_file is None:
            return self.base_filename + '.txt'
        else:
            return self._optuna_results_file

    @optuna_results_file.setter
    def optuna_results_file(self, value):
        self._optuna_results_file = value

    ######################################################################################################
    def run(self, random_state: int = 0):
        """
        Starts running the code, importing all modules that the program needs.
        """
        Util.check_and_create_directory(self.state.results_directory)

        logger.debug(f'training fraction = {self.data.train_size}')
        logger.debug(f'test_size = {(self.data.test_size + self.data.val_size)}')

        y = self.data.df_y.squeeze().values.reshape(-1, 1)
        X = self.data.df_X.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.data.train_size,
                                                            random_state=random_state)

        self.journal[C.TRAINING_POINTS] = X_train.shape[0]
        self.journal[C.TESTING_POINTS] = X_test.shape[0]
        self.journal[C.TRAIN_FRAC] = self.data.train_size
        self.journal[C.TEST_FRAC] = (self.data.test_size + self.data.val_size)

        # initial time
        t0 = time.time()
        # calling the function that performs the hyperparameter optimization to get the best network configuration
        best_params = self.opt_tune(X_train, y_train, X_test, y_test, )

        t_tune = time.time() - t0
        self.journal[C.RUN_TIME] = t_tune
