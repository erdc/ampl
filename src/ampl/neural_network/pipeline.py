from dataclasses import dataclass

from ampl.state import State
from ampl.neural_network.model_build import PipelineModelBuild
from ampl.neural_network.model_ensemble import PipelineModelEnsemble
from ampl.neural_network.model_eval import PipelineModelEval
from ampl.neural_network.optuna import PipelineOptuna
from ampl.neural_network.infer import PipelineModelInfer


@dataclass
class Pipeline(object):
    """
    Pipeline class which runs PipelineSteps.run()
    """
    state: State
    optuna: PipelineOptuna = None
    build: PipelineModelBuild = None
    eval: PipelineModelEval = None
    ensemble: PipelineModelEnsemble = None
    infer: PipelineModelInfer = None

    def run_all(self):
        if self.optuna:
            self.optuna.run()
        if self.build:
            self.build.run()
        if self.eval:
            self.eval.run()
        if self.ensemble:
            self.ensemble.run()
