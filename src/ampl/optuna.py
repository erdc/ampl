from typing import Literal

import pandas as pd
from optuna.trial import TrialState

from ampl.state import State
from ampl.pipelinestep import PipelineStep
from ampl.constant import Constant as C

import logging

logger = logging.getLogger(__name__)


class OptunaStep(PipelineStep):
    """

    """

    def __init__(self,
                 name: str,
                 state: State,
                 key: Literal[C.NN, C.DT, C.CNN]) -> None:
        """

        :type key: str
        """
        super().__init__(name, state, key)

    def process_optuna_study(self, study, n_top_trials: int = 5) -> list:
        """
        process optuna study results

        :param n_top_trials:
        :param study:
        :return:
        """
        # Run stats and show updated best trial
        # Grabbing the trial information

        pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
        failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]
        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        best_trial = study.best_trial

        # All parameters that are being varied by Optuna are added with this line:
        best_trial_df = pd.DataFrame.from_dict(best_trial.params.items())
        best_trial_df = best_trial_df.transpose()
        best_trial_df.rename(columns=best_trial_df.iloc[0], inplace=True)
        best_trial_df = best_trial_df.iloc[1:, :]
        best_trial_df = best_trial_df.convert_dtypes()

        study_df = study.trials_dataframe()
        completed_trials_df = study_df[study_df['state'] == 'COMPLETE'].sort_values(by=['value'])
        top_trials_df = completed_trials_df.head(n_top_trials).filter(regex='params')

        # removes the 'params_' prefix from the column names
        top_trials_df.columns = top_trials_df.columns.str.removeprefix('params_')

        best_trial_df = best_trial_df.reset_index(drop=True)
        top_trials_df = top_trials_df.reset_index(drop=True)
        int_cols = top_trials_df.columns[top_trials_df.columns.map(lambda x: x.startswith("n_"))]
        top_trials_df[int_cols] = top_trials_df[int_cols].fillna(0).astype(int)

        self.journal[C.N_TRIALS] = len(study.trials)

        self.journal[C.N_PRUNED_TRIALS] = len(pruned_trials)
        self.journal[C.N_FAILED_TRIALS] = len(failed_trials)
        self.journal[C.N_COMPLETED_TRIALS] = len(complete_trials)
        self.journal[C.BEST_TRIAL_RUN] = best_trial.value
        self.journal[C.BEST_TRIAL] = best_trial.params
        self.journal[C.TOP_TRIALS] = top_trials_df.to_dict(orient='records')

        best_params = list(best_trial.params.values())

        return best_params

    def run(self, random_state: int = 0):
        return NotImplementedError
