import os
import pickle

import numpy as np

from ampl.config import Configuration
from ampl.enums import NNOptimizers
from ampl.neural_network.util import Util as UtilNN

from pathlib import Path

import unittest


class TestPipelineModelBuild(unittest.TestCase):

    def test_get_keras_optimizer(self):
        optimizer = NNOptimizers.ADAMAX
        tf = UtilNN.load_tensorflow()
        self.assertTrue(isinstance(UtilNN.get_keras_optimizer(0.041, optimizer), tf.keras.optimizers.Adamax))


class TestBuild(unittest.TestCase):

    def setUp(self) -> None:
        this_path = Path(__file__).parent
        self.config_file = str(this_path / 'data/pipeline_config.yml')
        self.config = Configuration(self.config_file)
        self.pipeline = self.config.create_pipeline_nn()

        # os.remove(self.pipeline.optuna_nn.storage_log)

        self.pipeline.optuna.run()

        X_train, X_val, self.X_test, y_train, y_val, self.y_test = self.pipeline.state.data.train_val_test_split()

        self.pipeline.build.run()
        # self.pipeline.build_nn.run(overwrite_model=False)

        self.model = self.pipeline.build.models[0]

        tf = UtilNN.load_tensorflow()

        load_model_file = self.pipeline.state.get_model_path(1, '.keras')
        self.model.save(load_model_file)
        self.reconstructed_model = tf.keras.models.load_model(load_model_file)

    # def test_model_0(self):
    #     # Making sure our test acutally works
    #     print("Expected Model: ")
    #     self.expected_model.summary()
    #     print("Actual Model: ")
    #     self.model.summary()
    #     expected = self.expected_model.predict(self.X_test)
    #     actual = self.model.predict(self.X_test)
    #     np.testing.assert_allclose(expected, actual)

    def test_model_1(self):
        print("Reconstructed Model: ")
        self.reconstructed_model.summary()
        expected = self.model.predict(self.X_test)
        actual = self.reconstructed_model.predict(self.X_test)
        np.testing.assert_allclose(expected, actual)

    def test_model_2(self):
        # Evaluate the constructed model
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        print('\nConstructed model, accuracy: {:5.2f}%'.format(100 * acc), '\n')

        # Evaluate the reconstructed model
        re_loss, re_acc = self.reconstructed_model.evaluate(self.X_test, self.y_test, verbose=2)
        print('\nRestored model, accuracy: {:5.2f}%'.format(100 * re_acc), '\n')
        self.assertAlmostEqual(acc, re_acc, 3, 'Evaluating reconstructed model')
