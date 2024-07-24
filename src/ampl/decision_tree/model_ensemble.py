from dataclasses import dataclass, field

from ampl.enums import EnsembleMode
from ampl.state import State
from ampl.pipelinestep import PipelineStep
from ampl.constant import Constant as C

import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineModelEnsemble(PipelineStep):
    """
        Create the ensemble of good models to perform predictions with. This code requires models to be in
        the database to perform different ensemble methods.
        :param ensemble_mode:
        :param ensemble_results_file:
        :param callback:
        :type state: object
    """
    state: State
    ensemble_mode: EnsembleMode
    num_models: int
    num_points: int = field(default=None, init=False)

    def __post_init__(self):
        super().__init__(C.ENSEMBLE_DT, self.state, C.DT)
        self._ensemble_mode = self.ensemble_mode.value

    def run(self, random_state: int = 0 ):
        pass
