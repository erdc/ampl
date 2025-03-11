from dataclasses import dataclass, field

import pandas as pd

from ampl.state import State
from ampl.infer import ModelInfer
from ampl.constant import Constant as C
from ampl.neural_network.util import Util


import logging

logger = logging.getLogger(__name__)

@dataclass
class PipelineModelInfer(ModelInfer):
    """
        Makes inferences on new unseen data using an existing model. This code requires a saved model and a metadata
        file to be in the results directory. The metadata file is used to collect information for pre-processing the
        new data using the same schemes as the original data that the model was trained on.
        :param state:
        :type state: State
        :param callback:
        :type state: object
    """
    state: State
    infer_dataset_name: str
    df_inf: pd.DataFrame

    def __post_init__(self):
        super().__init__(C.INFER_NN, self.state, C.NN)

    def load_model(self, model_index: int, model_ext: str = C.MODEL_EXT_NN):
        load_model_file = self.state.get_model_path(model_index, model_ext=model_ext)
        tf = Util.load_tensorflow()
        model = tf.keras.models.load_model(load_model_file)
        logging.debug(model.summary())

        return model