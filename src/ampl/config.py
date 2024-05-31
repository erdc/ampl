from typing import Callable

import yaml
from jinja2 import Template, Undefined

from ampl.util import Util
from ampl.enums import *
from ampl.state import State
from ampl.data import Data, Database, read_csv, read_sql
from ampl.feature_importance import FeatureImportance
import ampl.neural_network.pipeline
import ampl.decision_tree.pipeline
from ampl.decision_tree.optuna_options import OptunaOptions, Range

import logging

logger = logging.getLogger(__name__)


class Configuration(dict):
    """
    Convenience function for simplified pipeline construction from yaml config file provided by the user.
    This yaml file will replace the default values that are used
    """

    def __init__(self, config_file: str, target_col_function: Callable = None) -> object:
        """
        :rtype: object
        :param target_col_function:
        :type config_file: object
        """
        self.config_file = config_file

        yaml.SafeLoader.add_constructor("!python/object:pruners.MedianPruner",
                                        Configuration.median_pruner_constructor)
        yaml.SafeLoader.add_constructor("!python/object:samplers.TPESampler",
                                        Configuration.sampler_constructor)
        yaml.SafeLoader.add_constructor("!python/object:tensorflow.keras.initializers.RandomUniform",
                                        Configuration.random_uniform_constructor)

        yaml.SafeLoader.add_constructor("!python/object:tensorflow.keras.initializers.GlorotUniform",
                                        Configuration.glorot_uniform_constructor)

        with open(self.config_file, 'r') as f:
            config_text = f.read()

        t = Template(config_text, undefined=Configuration.NullUndefined)
        c = yaml.safe_load(t.render())

        all_config = yaml.safe_load(t.render(c))

        super().__init__(all_config)

        self.data_db = None
        self.results_db = None
        self.data = None
        self.state = None
        self.target_col_function = target_col_function

        Util.check_and_create_directory(self['model']['results_directory'])

    @staticmethod
    def median_pruner_constructor(loader, node):
        from optuna.pruners import MedianPruner
        return MedianPruner()

    @staticmethod
    def sampler_constructor(loader, node):
        from optuna.samplers import TPESampler
        return TPESampler()

    @staticmethod
    def random_uniform_constructor(loader, node):
        import tensorflow.keras as keras

        kwargs = loader.construct_mapping(node, deep=True)['kwargs']
        return keras.initializers.RandomUniform(**kwargs)

    @staticmethod
    def glorot_uniform_constructor(loader, node):
        import tensorflow.keras as keras
        return keras.initializers.GlorotUniform()

    class NullUndefined(Undefined):
        """

        """
        ### https://stackoverflow.com/a/47706517/2152935
        def __getattr__(self, key):
            return ''

    def create_data_db(self) -> Database:
        if self.data_db is None:
            self.data_db = Database(self['db']['data_file'])
        return self.data_db

    def create_feature_importance(self) -> FeatureImportance:

        number_of_features = self['feature_importance']['number_of_features']
        feature_list = self['feature_importance']['feature_list']

        feature_importance = FeatureImportance(
            number_of_features=number_of_features,
            feature_list=feature_list,
        )

        return feature_importance

    def create_data(self) -> Data:
        if self.data is None:
            cols_to_enum = self['data']['cols_to_enum']
            target_variable = self['model']['target_variable']
            csv_file = self['data']['csv_file']
            # normalize_data = self['data']['normalize_data']
            table_name = self['data']['data_table_name']
            db = self['db']['data_file']
            feature_importance = self.create_feature_importance()
            train_size = self['data']['train_frac']
            remaining_size = (1.0-train_size)/2.0
            # if val_frac or test_frac not set then split the remaining size between them equally
            val_size = self['data'].get('val_frac', remaining_size)
            test_size = self['data'].get('test_frac', remaining_size)

            if csv_file:
                self.data = read_csv(csv_file,
                                     target_variable,
                                     cols_to_enum=cols_to_enum,
                                     feature_importance=feature_importance,
                                     # normalize_data=normalize_data,
                                     target_col_function=self.target_col_function,
                                     train_size=train_size,
                                     val_size=val_size,
                                     test_size=test_size
                                     )
            else:
                self.data = read_sql(table_name, db,
                                     target_variable,
                                     cols_to_enum=cols_to_enum,
                                     feature_importance=feature_importance,
                                     # normalize_data=normalize_data,
                                     target_col_function=self.target_col_function,
                                     train_size=train_size,
                                     val_size=val_size,
                                     test_size=test_size,
                                     )
        return self.data

    def _create_state(self) -> State:
        data = self.create_data()

        model_name = self['model']['model_name']

        study_name = self['model']['study_name']
        num_models = self['model']['num_models']
        target_variable = self['model']['target_variable']
        results_directory = self['model']['results_directory']
        saved_models_directory = self['model']['saved_models_directory']
        plots_directory = self['model']['plots_directory']
        number_of_features = self['feature_importance']['number_of_features']

        state = State(
            data,
            model_name,
            study_name,
            num_models,
            target_variable,
            number_of_features,
            results_directory=results_directory,
            saved_models_directory=saved_models_directory,
            plots_directory=plots_directory,
        )
        return state

    def create_state_nn(self) -> State:
        if self.state is None:
            self.state = self._create_state()
        self.state.top_trials_table_name = self['optuna_nn']['top_trials_table_name']
        self.state.best_trial_table_name = self['optuna_nn']['best_trial_table_name']

        return self.state

    def create_state_dt(self) -> State:
        if self.state is None:
            self.state = self._create_state()
        self.state.top_trials_table_name = self['optuna_dt']['top_trials_table_name']
        self.state.best_trial_table_name = self['optuna_dt']['best_trial_table_name']

        return self.state

    def create_optuna_nn(self) -> ampl.neural_network.PipelineOptuna:
        state = self.create_state_nn()

        n_trials = self['optuna']['n_trials']
        n_jobs = Util.get_n_jobs(percent=self['optuna']['parallel_percent'])
        direction = self['optuna']['direction']
        pruner = self['optuna']['pruner']

        loss = self['optuna_nn']['loss']
        trial_epochs = self['optuna_nn']['trial_epochs']
        trial_min_layers = self['optuna_nn']['trial_min_layers']
        trial_max_layers = self['optuna_nn']['trial_max_layers']
        max_neurons = self['optuna_nn']['max_neurons']
        min_lr = self['optuna_nn']['min_lr']
        max_lr = self['optuna_nn']['max_lr']
        initializer = self['optuna_nn']['initializer']
        activations = self['optuna_nn']['activations']
        optimizers = self['optuna_nn']['optimizers']

        opt_nn = ampl.neural_network.PipelineOptuna(state,
                                                    n_trials=n_trials,
                                                    trial_epochs=trial_epochs,
                                                    direction=Direction(direction),
                                                    pruner=pruner,
                                                    trial_min_layers=trial_min_layers,
                                                    trial_max_layers=trial_max_layers,
                                                    max_neurons=max_neurons,
                                                    min_lr=min_lr,
                                                    max_lr=max_lr,
                                                    activations=activations,
                                                    optimizers=optimizers,
                                                    loss=LossFunc(loss),
                                                    initializer=initializer,
                                                    n_jobs=n_jobs)

        return opt_nn

    def create_build_nn(self) -> ampl.neural_network.PipelineModelBuild:
        state = self.create_state_nn()

        loss = self['optuna_nn']['loss']
        epochs = self['nn']['epochs']
        patience = self['nn']['patience']

        build_nn = ampl.neural_network.PipelineModelBuild(state,
                                                          LossFunc(loss),
                                                          epochs=epochs,
                                                          patience=patience)

        return build_nn

    def create_eval_nn(self) -> ampl.neural_network.PipelineModelEval:
        state = self.create_state_nn()

        eval_nn = ampl.neural_network.PipelineModelEval(state)

        return eval_nn

    def create_ensemble_nn(self) -> ampl.neural_network.PipelineModelEnsemble:
        state = self.create_state_nn()

        ensemble_mode = self['ensemble_nn']['ensemble_mode']
        num_models = self['model']['num_models']

        ens_nn = ampl.neural_network.PipelineModelEnsemble(state,
                                                           EnsembleMode(ensemble_mode),
                                                           num_models)

        return ens_nn

    def create_pipeline_nn(self) -> ampl.neural_network.Pipeline:
        state = self.create_state_nn()
        optuna_nn = self.create_optuna_nn()
        build_nn = self.create_build_nn()
        eval_nn = self.create_eval_nn()
        ensemble_nn = self.create_ensemble_nn()

        _pipeline = ampl.neural_network.Pipeline(state,
                                                 build=build_nn,
                                                 eval=eval_nn,
                                                 ensemble=ensemble_nn,
                                                 optuna=optuna_nn
                                                 )

        return _pipeline

    def create_pipeline_dt(self) -> ampl.decision_tree.Pipeline:
        state = self.create_state_dt()
        optuna_dt = self.create_optuna_dt()
        build_dt = self.create_build_dt()
        eval_dt = self.create_eval_dt()
        ensemble_dt = self.create_ensemble_dt()

        _pipeline = ampl.decision_tree.Pipeline(state,
                                                optuna=optuna_dt,
                                                build=build_dt,
                                                eval=eval_dt,
                                                ensemble=ensemble_dt,
                                                )

        return _pipeline

    def create_optuna_dt(self):
        state = self.create_state_dt()

        loss = self['dt']['loss']
        eval_metric = self['dt']['eval_metric']

        n_trials = self['optuna']['n_trials']
        n_jobs = Util.get_n_jobs(percent=self['optuna']['parallel_percent'])
        direction = self['optuna']['direction']
        pruner = self['optuna']['pruner']

        early_stopping_rounds = self['optuna_dt']['early_stopping_rounds']
        observation_key = self['optuna_dt']['observation_key']
        sampler = self['optuna_dt']['sampler']
        multivariate = self['optuna_dt']['multivariate']

        r_vals = self['optuna_dt']['ranges']
        ranges = OptunaOptions()
        ranges.n_estimators = Range(*r_vals['n_estimators'])
        ranges.max_depth = Range(*r_vals['max_depth'])
        ranges.learning_rate = Range(*r_vals['learning_rate'])
        ranges.col_sample_by_tree = Range(*r_vals['col_sample_by_tree'])
        ranges.subsample = Range(*r_vals['subsample'])
        ranges.alpha = Range(*r_vals['alpha'])
        ranges.lambda_ = Range(*r_vals['lambda'])
        ranges.gamma = Range(*r_vals['gamma'])
        ranges.min_child_weight = Range(*r_vals['min_child_weight'])

        opt_nn = ampl.decision_tree.PipelineOptuna(state,
                                                   n_trials=n_trials,
                                                   loss=loss,
                                                   eval_metric=eval_metric,
                                                   early_stopping_rounds=early_stopping_rounds,
                                                   observation_key=observation_key,
                                                   direction=Direction(direction),
                                                   pruner=pruner,
                                                   sampler=sampler,
                                                   multivariate=multivariate,
                                                   n_jobs=n_jobs
                                                   )
        return opt_nn

    def create_build_dt(self):
        state = self.create_state_dt()

        loss = self['dt']['loss']
        eval_metric = self['dt']['eval_metric']
        early_stopping_rounds = self['dt']['early_stopping_rounds']
        verbosity = self['dt']['verbosity']

        build_nn = ampl.decision_tree.PipelineModelBuild(state,
                                                         loss=loss,
                                                         eval_metric=eval_metric,
                                                         early_stopping_rounds=early_stopping_rounds,
                                                         verbosity=verbosity)

        return build_nn

    def create_eval_dt(self):
        state = self.create_state_dt()

        eval_nn = ampl.decision_tree.PipelineModelEval(state)

        return eval_nn

    def create_ensemble_dt(self):
        state = self.create_state_dt()

        ensemble_mode = self['ensemble_dt']['ensemble_mode']
        num_models = self['model']['num_models']

        ens_nn = ampl.decision_tree.PipelineModelEnsemble(state,
                                                          EnsembleMode(ensemble_mode),
                                                          num_models)
        return ens_nn
