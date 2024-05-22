from ampl.config import Configuration
from ampl.state import State


from pathlib import Path

from pandas.testing import assert_frame_equal
import unittest


class TestState(unittest.TestCase):
    def setUp(self) -> None:
        this_path = Path(__file__).parent
        self.config_file = str(this_path / 'data/pipeline_config.yml')
        self.config1 = Configuration(self.config_file)
        self.full_pipeline = self.config1.create_pipeline_nn()

        self.config2 = Configuration(self.config_file)
        self.step_pipeline = self.config2.create_pipeline_nn()

    def test_load_trials_sql(self):
        state = self.config1.create_state_dt()
        self.assertTrue(isinstance(state, State))

    @staticmethod
    def compare_dataframe(left, right):
        fp_best_trial_df, fp_top_trials_df = left.build.load_trials_df()
        sp_best_trial_df, sp_top_trials_df = right.build.load_trials_df()
        assert_frame_equal(fp_best_trial_df, sp_best_trial_df, check_dtype=False)
        assert_frame_equal(fp_top_trials_df, sp_top_trials_df, check_dtype=False)

    def test_steps_state_1(self):
        assert_frame_equal(self.full_pipeline.state.data.feature_importance.results_df,
                           self.step_pipeline.state.data.feature_importance.results_df)
        self.assertEqual(self.full_pipeline.state, self.step_pipeline.state)
        fp_best_trial_df, fp_top_trials_df = self.full_pipeline.build.load_trials_df()
        sp_best_trial_df, sp_top_trials_df = self.step_pipeline.build.load_trials_df()
        self.assertIsNotNone(fp_best_trial_df)
        self.assertIsNotNone(fp_top_trials_df)
        self.assertIsNotNone(sp_best_trial_df)
        self.assertIsNotNone(sp_top_trials_df)

    def test_steps_state_2(self):
        self.full_pipeline.optuna.run()

        self.assertEqual(self.full_pipeline.state, self.step_pipeline.state)
        self.compare_dataframe(self.full_pipeline, self.step_pipeline)

    def test_steps_state_3(self):
        self.step_pipeline.build.run()
        self.assertEqual(self.full_pipeline.state, self.step_pipeline.state)
        self.compare_dataframe(self.full_pipeline, self.step_pipeline)
