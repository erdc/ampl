import unittest
from pathlib import Path

from ampl.config import Configuration
from ampl.data import *
from ampl.state import State
from ampl.neural_network import *


class TestConfigFile(unittest.TestCase):
    def setUp(self):
        this_path = Path(__file__).parent
        config_file = str(this_path / 'data/pipeline_config.yml')
        self.config = Configuration(config_file)
        self.config['model']['target_col_function'] = lambda df_: df_['x04'].astype('float') ** 2 - \
                                                                  df_['x05'].astype('float') ** 2

    def test_config_file(self):
        self.assertTrue(self.config['model']['dataset_name'], 'test')
        self.assertTrue(self.config['model']['study_name'], 'Ballistics')
        self.assertTrue(self.config['model']['target_variable'], 'KER')
        self.assertTrue(self.config['model']['dataset_name'], 'test')
        self.assertTrue(self.config['model']['model_name'], 'KER_test')
        self.assertTrue(self.config['model']['num_models'], 5)

    def test_create_database(self):
        db = self.config.create_data_db()
        self.assertTrue(isinstance(db, Database))
        assert True
        # self.assertTrue(data.df.shape, ())

    def test_create_dataframe(self):
        data = self.config.create_data()
        self.assertTrue(isinstance(data, Data))

        assert True
        # self.assertTrue(data.df.shape, ())

    def test_create_pipeline(self):
        pipeline = self.config.create_pipeline_nn()
        self.assertTrue(isinstance(pipeline, Pipeline))
        # self.assertEqual(pipeline.study_name, 'Ballistics')
        # self.assertEqual(pipeline.target_variable, 'KER')
        # self.assertEqual(pipeline.dataset_name, 'test')
        # self.assertEqual(pipeline.model_name, f'{pipeline.target_variable}_{pipeline.dataset_name}')

    def test_create_feature_importance(self):
        fi = self.config.create_feature_importance()
        self.assertTrue(isinstance(fi, FeatureImportance))

    def test_create_state_nn(self):
        model = self.config.create_state_nn()
        self.assertTrue(isinstance(model, State))

    def test_create_state_dt(self):
        model = self.config.create_state_dt()
        self.assertTrue(isinstance(model, State))

    def test_create_optuna_nn(self):
        opt_nn = self.config.create_optuna_nn()
        self.assertTrue(isinstance(opt_nn, PipelineOptuna))

    def test_create_build_nn(self):
        build_nn = self.config.create_build_nn()
        self.assertTrue(isinstance(build_nn, PipelineModelBuild))

    def test_create_eval_nn(self):
        eval_nn = self.config.create_eval_nn()
        self.assertTrue(isinstance(eval_nn, PipelineModelEval))

    def test_create_ensemble_nn(self):
        ens_nn = self.config.create_ensemble_nn()
        self.assertTrue(isinstance(ens_nn, PipelineModelEnsemble))
