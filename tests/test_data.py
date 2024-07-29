import math
import unittest
from pathlib import Path
from ampl.config import Configuration
from ampl.data import *
from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_equal


class TestDatabase(unittest.TestCase):
    sql_data_file = './data/dataset_temp.sqlite'
    db = Database(sql_data_file)

    def test_connection(self):
        with self.db.connection() as conn:
            self.assertIsNotNone(conn.cursor())

    def test_connect(self):
        with Database.connect(self.sql_data_file) as conn:
            self.assertIsNotNone(conn.cursor())


class TestData(unittest.TestCase):

    def setUp(self) -> None:
        this_path = Path(__file__).parent
        self.config_file = str(this_path / 'data/pipeline_config.yml')
        self.target_func = lambda df_: (df_['x04'].astype('float') ** 2 -
                                        df_['x05'].astype('float') ** 2)

        self.config = Configuration(self.config_file, target_col_function=self.target_func)

    def test_load_data(self):
        db = self.config.create_data_db()
        with self.assertRaises(AttributeError) as cm:
            data = read_csv('./data/dataset.csv', None)

    def test_Data(self):
        with self.assertRaises(ValueError) as cm:
            data = Data(None, '')

    def test_dataframe(self):
        data = self.config.create_data()

        self.assertIsNotNone(data.df)

        csv_file = './data/dataset.csv'
        self.assertTrue(Path.is_file(Path(csv_file)))

        with self.assertRaises(ValueError) as cm:
            data2 = read_csv(csv_file, 'y', target_col_function=self.target_func,
                             feature_importance=self.config.create_feature_importance())

        data3 = read_csv(csv_file, 'y', target_col_function=self.target_func,
                         cols_to_enum=self.config['data']['cols_to_enum'],
                         feature_importance=self.config.create_feature_importance())

        this_path = Path(__file__).parent
        db_file = str(this_path / 'data/temp.sqlite')
        with Database.connect(db_file) as conn:
            data3.df.to_sql('dataset', conn, if_exists='replace', index=False)
        self.assertTrue(Path.is_file(Path(db_file)))

    def test_init(self):
        db = self.config.create_data_db()
        # with self.assertRaises(ValueError) as cm:
        #     df = Data(db, self.config['data']['sql_table_name'], 'KER')
        #
        # with self.assertRaises(ValueError) as cm:
        #     df = Data(db, self.config['data']['sql_table_name'], 'KER',
        #                 load_from_preprocessed_table=True)
        #
        # with self.assertRaises(FileNotFoundError) as cm:
        #     df = Data(db, self.config['data']['sql_table_name'], 'KER',
        #                 csv_file='junk.csv')


class TestData2(unittest.TestCase):

    def setUp(self) -> None:
        self.this_path = Path(__file__).parent
        self.config_file = str(self.this_path / 'data/pipeline_config_2.yml')
        self.config = Configuration(self.config_file)

    def test_load_data1(self):
        data1 = read_sql(self.config['data']['data_table_name'], self.config['db']['data_file'],
                         self.config['model']['target_variable'])
        if self.config['data']['data_normalized_table_name'] is not None:
            # write the preprocessed DataFrame to a SQLite database
            SqlUtil.to_sql(data1.df, self.config.create_data_db(),
                           self.config['data']['data_normalized_table_name'], if_exists='replace')

        # data2 = ampl.read_sql(self.config['data']['sql_preprocessed_table_name'], self.config['db']['sql_data_file'],
        #                       self.config['model']['target_variable'])

    def test_load_data2(self):
        config_file = str(self.this_path / 'data/pipeline_config.yml')
        config1 = Configuration(config_file)
        full_pipeline = config1.create_pipeline_nn()

        config2 = Configuration(config_file)
        step_pipeline = config2.create_pipeline_nn()
        self.assertEqual(full_pipeline.state.data.feature_importance, step_pipeline.state.data.feature_importance)
        assert_frame_equal(full_pipeline.state.data.feature_importance.results_df,
                           step_pipeline.state.data.feature_importance.results_df)
        assert_array_equal(full_pipeline.state.data.feature_importance.shap_values,
                           step_pipeline.state.data.feature_importance.shap_values)

        self.assertEqual(full_pipeline.state.data, step_pipeline.state.data)
        assert_frame_equal(full_pipeline.state.data.df, step_pipeline.state.data.df)

        full_pipeline.optuna.run()

        self.assertEqual(full_pipeline.state.data, step_pipeline.state.data)

        step_pipeline.build.run()

        self.assertEqual(full_pipeline.state.data, step_pipeline.state.data)


class TestData3(unittest.TestCase):
    def test_test_train_split(self):
        this_path = Path(__file__).parent
        config_file = str(this_path / 'data/pipeline_config.yml')
        config = Configuration(config_file)

        data = config.create_data()
        n_size = data.df.shape[0]
        self.assertEqual(n_size, 25870)
        val_size = 0.15
        test_size = 0.15
        X_train, X_val, X_test, y_train, y_val, y_test = data.train_val_test_split(val_size=val_size,
                                                                                   test_size=test_size)
        self.assertAlmostEqual(len(X_train), int(n_size * (1 - (val_size + test_size))), delta=1)
        self.assertAlmostEqual(len(X_val), int(n_size * val_size), delta=1)
        self.assertAlmostEqual(len(X_test), int(n_size * test_size), delta=1)


class TestPreSplitData(unittest.TestCase):
    def test_df(self):
        this_path = Path(__file__).parent
        config_file = str(this_path / 'Brianna/ampl_config.yml')
        config = Configuration(config_file)

        data = config.create_data()
        self.assertTrue(True)
