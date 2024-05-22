import unittest
from pathlib import Path
import pandas as pd
from ampl.feature_importance import FeatureImportance
from ampl.config import Configuration
from ampl.data import Data


class TestFeatureImportance(unittest.TestCase):

    def setUp(self) -> None:
        self.this_path = Path(__file__).parent
        config_file = str(self.this_path / 'data/pipeline_config.yml')
        self.config = Configuration(config_file)
        self.fi = self.config.create_feature_importance()

    def test_init(self):
        fi = self.config.create_feature_importance()
        self.assertTrue(isinstance(fi, FeatureImportance))

    def test_run(self):
        df = pd.read_csv('./data/dataset+target.csv')
        target = 'KER'

        df_y = df[[target]]
        df_X = df.drop(columns=[target])

        n_features = self.config['feature_importance']['number_of_features']
        feature_list = self.config['feature_importance']['feature_list']

        df_y = df_y.astype(float)
        cols_to_enum = ['Nose_Shape', 'Proj_Matl', 'Target_Matl']
        for col in cols_to_enum:
            df_X[col] = df_X[col].astype('category').cat.codes
        df_o = df_X.select_dtypes(include='object')
        df_X[df_o.columns] = df_X[df_o.columns].astype('float')

        self.assertNotEqual(len(df_X.columns), n_features)

        actual_fi_cols = self.fi.run(df_X, df_y)
        self.assertEqual(len(actual_fi_cols), n_features)
