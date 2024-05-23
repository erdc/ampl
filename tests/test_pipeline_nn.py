from ampl.config import Configuration
from ampl.neural_network import Pipeline
import os
from pathlib import Path

import unittest


class TestPipelineNN(unittest.TestCase):
    def setUp(self) -> None:
        this_path = Path(__file__).parent
        config_file = str(this_path / 'data/pipeline_config.yml')
        self.config = Configuration(config_file)
        self.config['model']['target_col_function'] = \
            lambda df_: df_['x04'].astype('float') ** 2 - \
                        df_['x05'].astype('float') ** 2

        self.pipeline = self.config.create_pipeline_nn()

    def test_step_1_pipeline(self):
        # self.pipeline = self.config.create_pipeline()
        self.assertTrue(self.pipeline is not None)

        self.assertTrue(isinstance(self.pipeline, Pipeline))
        self.assertEqual(self.pipeline.state.data.df_X.shape[1], self.pipeline.state.number_of_features)

    def test_step_2_optuna_nn(self):
        self.pipeline.optuna.run()
        self.assertTrue(True, 'Pipeline Optuna NN run')

    def test_step_3_build_nn(self):
        self.pipeline.build.run()
        self.assertTrue(True, 'Pipeline Build NN run')

    def test_step_4_eval_nn(self):
        self.pipeline.eval.run()
        self.assertTrue(True, 'Pipeline Eval NN run')

    def test_step_5_ensemble_nn(self):
        self.pipeline.ensemble.run()
        self.assertTrue(True, 'Pipeline Ensemble NN run')

