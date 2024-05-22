"""
AMPL - Automated Machine-learning Pipeline Package

Use this module like this:

.. code-block:: python

    filename = 'data/ampl_config.yml'

    # Create default config file
    Util.create_default_config_file(filename)

    # Update config file with run data and information

    config = Configuration(filename)

    pipeline = config.create_pipeline_nn()
    pipeline.run_all()


Comprehensive Usage:

.. code-block:: python

    import ampl

    config = ampl.Configuration('data/ampl_config.yml')

    # Creating Neural Network AMPL
    pipeline_nn = config.create_pipeline_nn()

    pipeline_nn.optuna.run()
    # Getting Best Trial and Top Trials from Optuna Study
    best_trial_df, top_trials_df = pipeline_nn.optuna.load_trials_df()
    print(best_trial_df)
    print(top_trials_df)

    pipeline_nn.build.run()
    pipeline_nn.eval.run()
    pipeline_nn.ensemble.run()

    # # Creating Decision Tree AMPL
    pipeline_dt = config.create_pipeline_dt()

    pipeline_dt.optuna.run()
    pipeline_dt.build.run()
    pipeline_dt.eval.run()
    pipeline_dt.ensemble.run()

AMPL Module UML Diagram

.. image:: ./images/classes_AMPL.png

"""

from ampl.util import *
from ampl.enums import *
from ampl.constant import Constant
from ampl.cli import PipelineCli
from ampl.config import Configuration
from ampl.feature_importance import FeatureImportance
from ampl.data import *
from ampl.state import State

