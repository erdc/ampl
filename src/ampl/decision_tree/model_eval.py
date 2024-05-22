from dataclasses import dataclass
from xgboost import XGBRegressor

from ampl.constant import Constant as C

from ampl.eval import ModelEval
from ampl.state import State

import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineModelEval(ModelEval):
    """
        :param state:
        :type state: State
    """
    state: State

    def __post_init__(self):
        super().__init__(C.EVALUATE_DT, self.state, C.OPTUNA_DT)

    def load_model(self, model_index: int):
        logging.debug('Running DT Eval step')

        load_model_file = self.state.get_model_path(model_index, model_ext=C.MODEL_EXT_DT)
        model = XGBRegressor()
        model.load_model(load_model_file)

        return model

    def custom_plots(self, model_index: int):
        pass
