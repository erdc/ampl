from ampl.config import Configuration
from ampl.decision_tree import Pipeline
import os
from pathlib import Path

import unittest


class TestPipelineDT(unittest.TestCase):
    def setUp(self) -> None:
        this_path = Path(__file__).parent
        config_file = str(this_path / 'data/pipeline_config.yml')
        self.config = Configuration(config_file)
        target_function = lambda df_: df_['x04'].astype('float') ** 2 - df_['x05'].astype('float') ** 2
        self.config = Configuration(config_file, target_col_function=target_function)


        self.pipeline = self.config.create_pipeline_dt()

    def test_step_1_pipeline_dt(self):
        # self.pipeline = self.config.create_pipeline()
        self.assertTrue(self.pipeline is not None)

        self.assertTrue(isinstance(self.pipeline, Pipeline))
        self.assertEqual(self.pipeline.state.data.df_X.shape[1], self.pipeline.state.number_of_features)

    def test_step_2_optuna_dt(self):
        self.pipeline.optuna.run()
        self.assertTrue(True, 'Pipeline Optuna DT run')

    def test_step_3_build_dt(self):
        self.pipeline.build.run()
        self.assertTrue(True, 'Pipeline Build DT run')

    def test_step_4_eval_dt(self):
        self.pipeline.eval.run()
        self.assertTrue(True, 'Pipeline Eval DT run')

    def test_step_5_ensemble_dt(self):
        self.pipeline.ensemble.run()
        self.assertTrue(True, 'Pipeline Ensemble DT run')

