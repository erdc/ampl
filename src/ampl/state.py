from dataclasses import dataclass, field, asdict

import shutil
from ampl.util import Util
from ampl.data import Data, Database
from ampl.constant import Constant as C

import logging

logger = logging.getLogger(__name__)


@dataclass
class State:
    """
    The common pipeline state class which hold all variables needed for any step in the auto machine learning pipeline.
    """
    data: Data
    model_name: str
    study_name: str
    num_models: int
    target_variable: str
    number_of_features: int
    infer_dataset_name: str
    results_directory: str = './results/'
    saved_models_directory: str = './results/saved_models/'
    plots_directory: str = './results/plots/'
    journal_file: str = None
    metadata_file: str = None
    std_perc_err: float = field(default=None, init=False)

    def __post_init__(self):
        self.results_directory = Util.add_slash(self.results_directory)
        self.saved_models_directory = Util.add_slash(self.saved_models_directory)
        self.plots_directory = Util.add_slash(self.plots_directory)

        if self.journal_file is None:
            self.journal_file = (self.results_directory + 'journal_' + self.model_name + '.json')

        Util.check_and_create_directory(self.results_directory)
        Util.check_and_create_directory(self.saved_models_directory)
        Util.check_and_create_directory(self.plots_directory)

        self.data.feature_importance.results_df.to_csv(
                    self.results_directory + self.data.feature_importance.name + '_' + self.model_name + '.csv',
                    index=False, header=True)

        journal = dict()
        journal[C.MODEL_NAME] = self.model_name
        journal[C.STUDY_NAME] = self.study_name
        journal[C.N_MODELS] = self.num_models
        journal[C.TARGET_VARIABLE] = self.target_variable
        journal[C.N_FEATURES] = self.number_of_features
        journal[C.TOTAL_POINTS] = self.data.df.shape[0]
        journal[C.RESULTS_DIR] = self.results_directory
        journal[C.SAVED_MODELS_DIR] = self.saved_models_directory
        journal[C.PLOTS_DIR] = self.plots_directory
        journal[C.METADATA_FILE] = self.results_directory + C.METADATA_JSON
        journal[C.DATA] = asdict(self.data)

        # removing dataframe from journal
        journal[C.DATA].pop(f"{self.data.df=}".split("=")[0].split('.')[-1])  # removing dataframe from journal
        journal[C.DATA].pop(f"{self.data.target_col_function=}".split("=")[0].split('.')[-1])  # removing from journal
        journal[C.DATA].pop(f"{self.data.feature_importance=}".split("=")[0].split('.')[-1])  # removing from journal
        journal[C.DATA].pop(f"{self.data.encoders=}".split("=")[0].split('.')[-1])  # removing from journal
        journal[C.DATA].pop(f"{self.data.encoder_mapping=}".split("=")[0].split('.')[-1])  # removing from journal
        journal[C.DATA].pop(f"{self.data.column_stats=}".split("=")[0].split('.')[-1])  # removing from journal

        journal[C.DATA][C.FEATURE_IMPORTANCE] = asdict(self.data.feature_importance)
        journal[C.DATA][C.FEATURE_IMPORTANCE].pop(f"{self.data.feature_importance.results_df=}".split("=")[0].split('.')[-1])
        journal[C.DATA][C.FEATURE_IMPORTANCE][C.RESULTS] = self.data.feature_importance.results_df.to_dict()
        journal[C.DATA][C.FEATURE_IMPORTANCE].pop(f"{self.data.feature_importance.shap_values=}".split("=")[0].split('.')[-1])

        if self.metadata_file is None:
            self.metadata_file = (self.results_directory + 'metadata_' + self.model_name + '.json')

        metadata = {}
        metadata['journal_file'] = self.journal_file
        metadata['order_of_model_features'] = self.data.feature_importance_list
        metadata['column_stats'] = self.data.column_stats
        if self.data.cols_to_enum:
            metadata['encoder_mapping'] = self.data.encoder_mapping
        else:
            metadata['encoder_mapping'] = {}
        Util.write_dict_to_json(metadata, self.metadata_file)

        Util.update_journal(C.MODEL, journal, self.journal_file)

    def get_model_base_path(self, model_number: int) -> str:
        file_base = self.saved_models_directory + self.model_name + '_top_' + str(model_number)

        return file_base

    def get_model_path(self, model_number: int, model_ext: str):
        """
        Sets the path for loading the model and history for model evaluation.
        """
        file_base = self.get_model_base_path(model_number)
        if model_ext[0] != '.':
            model_ext = '.' + model_ext
        load_model_file = file_base + '_model' + model_ext

        return load_model_file


