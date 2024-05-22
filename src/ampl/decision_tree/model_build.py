import pickle
import time
from dataclasses import dataclass

from xgboost import XGBRegressor

from ampl.enums import Verbosity
from ampl.state import State
from ampl.pipelinestep import PipelineStep
from ampl.constant import Constant as C
from ampl.util import Util

import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineModelBuild(PipelineStep):
    """

    """
    state: State
    loss: str = "reg:squarederror"
    ''' Learning task's learning objective https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters'''
    eval_metric: str = 'rmse'
    ''' Learning task's eval_metric https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters'''
    early_stopping_rounds: int = 200
    verbosity: Verbosity = Verbosity.Info

    def __post_init__(self):
        super().__init__(C.BUILD_DT, self.state, C.OPTUNA_DT)

    def run(self, random_state: int = 0, model_ext=C.MODEL_EXT_DT):

        Util.check_and_create_directory(self.state.results_directory)
        Util.check_and_create_directory(self.state.saved_models_directory)

        X_train, X_val, X_test, y_train, y_val, y_test = self.state.data.train_val_test_split(random_state=random_state)

        self.journal[C.TRAINING_POINTS] = X_train.shape[0]
        self.journal[C.TESTING_POINTS] = X_test.shape[0]
        self.journal[C.VALIDATION_POINTS] = X_val.shape[0]

        ################################## Model Training ######################
        best_trial_df, top_trials_df = self.load_trials_df()
        j = 0
        assert len(best_trial_df) > j, 'Optuna Best Trial is not found/loaded'

        hp = best_trial_df.iloc[j].to_dict()
        hp["objective"] = self.loss
        hp["verbosity"] = self.verbosity
        hp["n_estimators"] = int(hp['n_estimators'])        
        hp['max_depth'] = int(hp['max_depth'])
        # hp['eval_metric'] = self.eval_metric
        # hp['early_stopping_rounds'] = self.early_stopping_rounds,

        self.journal[C.DT_PARAMS] = hp

        ####################### Train Model #######################
        t_start = time.time()

        model = XGBRegressor(**hp)
        history = model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_test, y_test)],
                      eval_metric=self.eval_metric,
                      verbose=0,
                      early_stopping_rounds=self.early_stopping_rounds)

        #################################### Model Save ################################
        self.state.saved_model_file = self.state.get_model_path(j, model_ext=model_ext)

        model.save_model(self.state.saved_model_file)

        self.journal[C.RUN_TIME] = time.time() - t_start
