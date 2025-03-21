import abc
from typing import Any, Literal

import numpy as np

from ampl.util import Util
from ampl.state import State
from ampl.pipelinestep import PipelineStep
from ampl.constant import Constant as C

import logging
import pandas as pd
import json

logger = logging.getLogger(__name__)

class ModelInfer(PipelineStep):
    def __init__(self,
                 name: str,
                 state: State,
                 key: Literal[C.NN, C.DT, C.CNN]) -> None:
        super().__init__(name, state, key)

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'load_model') and
                callable(subclass.load_model)
                or
                NotImplemented)

    @abc.abstractmethod
    def load_model(self, model_index: int) -> Any:
        raise NotImplementedError

    def run(self, random_state: int = 0):
        logging.debug('Running NN Inferencing step')
        Util.check_and_create_directory(self.state.results_directory)
        Util.check_and_create_directory(self.state.saved_models_directory)
        Util.check_and_create_directory(self.state.plots_directory)

        # Load metadata file
        with open(self.state.metadata_file, 'r') as file:
            metadata = json.load(file)

        # Load necessary data from metadata file for pre-processing
        features = metadata.get('order_of_model_features')
        encoders = metadata.get('encoder_mapping')
        col_stats = metadata.get('column_stats')

        # Get target variable stats necessary for de-normalizing
        target_min = col_stats.get(self.data.target_variable, {}).get('min')
        target_max = col_stats.get(self.data.target_variable, {}).get('max')

        X = self.df_inf[features]
        predictions = X.copy()

        # Encode new categorical data if necessary
        if encoders:
            for col, mapping in encoders.items():
                if col in X.columns:
                    X[col] = X[col].map(mapping)

        # Normalize new data using original data normalization scheme
        for col in X:
            minimum = col_stats.get(col, {}).get('min')
            maximum = col_stats.get(col, {}).get('max')
            X[col] = X[col].apply(lambda x: (x - minimum) / (maximum - minimum))

        for j in range(self.state.num_models):
            model = self.load_model(j)

            # Make predictions on new data and de-normalize predictions
            X_pred = model.predict(X)  # Make predictions on new data with existing model
            X_pred = np.reshape(X_pred, (-1, 1))
            X_pred = X_pred * (target_max - target_min) + target_min  # De-normalize predictions

            # Print and save predictions
            predictions[self.data.target_variable + '_pred'] = X_pred
            predictions.to_csv(
                self.state.results_directory + 'model_inferencing_' + self.state.model_name + '_'
                + self.state.infer_dataset_name + '_' + str(j) + C.CSV_EXT,
                index=False)



