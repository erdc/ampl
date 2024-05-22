import unittest
from pathlib import Path

from ampl.config import Configuration


class TestNN(unittest.TestCase):
    """
    Test Individual Steps of the Neural Network Pipeline
    """

    def setUp(self):
        this_path = Path(__file__).parent
        self.config_file = str(this_path / 'data/pipeline_config.yml')

    def test_step_1_optuna(self):
        # TODO Add checks for result file
        config = Configuration(self.config_file)

        optuna_nn = config.create_optuna_nn()
        optuna_nn.run()

    def test_step_2_build_nn(self):
        config = Configuration(self.config_file)

        build_nn = config.create_build_nn()
        build_nn.run()

    def test_step_3_eval_nn(self):
        config = Configuration(self.config_file)

        eval_nn = config.create_eval_nn()
        eval_nn.run()

    def test_step_4_ensemble_nn(self):
        config = Configuration(self.config_file)

        ensemble_nn = config.create_ensemble_nn()
        ensemble_nn.run()
