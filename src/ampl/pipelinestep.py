import abc
import os
import json
from typing import Literal

import pandas as pd

from ampl.util import Util
from ampl.state import State

from ampl.constant import Constant as C


class PipelineStep(object):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'run') and
                callable(subclass.run)
                or
                NotImplemented)

    def __init__(self,
                 name: str,
                 state: State,
                 key: Literal[C.OPTUNA_NN, C.OPTUNA_DT, C.OPTUNA_CNN]) -> None:
        """
            Create the ensemble of good models to perform predictions with. This code requires models to be in
            the database to perform different ensemble methods.
            :rtype: object
            :param state:
            :type state: object
        """
        self.name = name
        self.state = state
        self.key = key
        self.journal = dict()

        # wrapping the run method with persist_journal
        self.run = Util.persist_journal(self.run, self.name, self.journal, self.state.journal_file)

    @property
    def data(self):
        return self.state.data

    @property
    def db(self):
        return self.state.results_db

    @property
    def base_filename(self):
        return self.state.results_directory + '/' + self.name + '_' + self.state.model_name

    @abc.abstractmethod
    def run(self, random_state: int = 0):
        raise NotImplementedError

    def load_trials_df(self):
        """
        Reads the JSON file into a dictionary for loading the best trial and top trials with the
        optimal neural network configuration from the Optuna search.
        """
        full_journal = dict()
        if os.path.exists(self.state.journal_file):
            with open(self.state.journal_file, 'r') as fp:
                full_journal = json.load(fp)

        best_trial_df = None
        top_trials_df = None

        if self.key in full_journal:
            best_trial_df = pd.DataFrame.from_records([full_journal[self.key].get(C.BEST_TRIAL)])
            top_trials_df = pd.DataFrame.from_records(full_journal[self.key].get(C.TOP_TRIALS))

        return best_trial_df, top_trials_df



