import abc
from typing import Any, Literal

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error

from ampl.util import Util
from ampl.state import State
from ampl.pipelinestep import PipelineStep
from ampl.constant import Constant as C

import logging

logger = logging.getLogger(__name__)


class ModelEval(PipelineStep):
    def __init__(self,
                 name: str,
                 state: State,
                 key: Literal[C.OPTUNA_NN, C.OPTUNA_DT]) -> None:
        super().__init__(name, state, key)

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'load_model') and
                callable(subclass.load_model)
                or
                NotImplemented)

    @abc.abstractmethod
    def load_model(self, model_index: int) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def custom_plots(self, model_index: int):
        raise NotImplementedError

    def run(self, random_state: int = 0):
        logging.debug('Running NN Eval step')
        Util.check_and_create_directory(self.state.results_directory)
        Util.check_and_create_directory(self.state.saved_models_directory)
        Util.check_and_create_directory(self.state.plots_directory)

        X_train, X_val, X_test, y_train, y_val, y_test = self.state.data.train_val_test_split(random_state=random_state)

        input_shape = (X_train.shape[1], )
        logging.debug(f'{input_shape = }')

        self.journal[C.TRAINING_POINTS] = X_train.shape[0]
        self.journal[C.TESTING_POINTS] = X_test.shape[0]
        self.journal[C.VALIDATION_POINTS] = X_val.shape[0]

        y_train = self.data.denormalize_y(y_train)
        y_test = self.data.denormalize_y(y_test)
        y_val = self.data.denormalize_y(y_val)

        logging.debug(f'{self.state.num_models = }')
        self.journal[C.N_MODELS] = self.state.num_models

        for j in range(self.state.num_models):
            results = dict()
            self.journal[f'model_{j}'] = results

            model = self.load_model(j)

            # Make Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_val_pred = model.predict(X_val)

            y_train_pred = np.reshape(y_train_pred, (-1, 1))
            y_test_pred = np.reshape(y_test_pred, (-1, 1))
            y_val_pred = np.reshape(y_val_pred, (-1, 1))

            y_train_pred = self.data.denormalize_y(y_train_pred)
            y_test_pred = self.data.denormalize_y(y_test_pred)
            y_val_pred = self.data.denormalize_y(y_val_pred)

            # Mean Absolute Error
            mae = mean_absolute_error(y_val, y_val_pred)

            # Mean Square Error
            mse = mean_squared_error(y_val, y_val_pred)

            # Root Mean Square Error
            rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

            # R²
            r2 = r2_score(y_val, y_val_pred)  # Best possible score is 1.0, lower values are worse

            # Normalized Root Mean Squared Error
            nrmse = rmse / (y_val.max() - y_val.min())

            # Max Error
            max_err = max_error(y_val, y_val_pred)

            # MAPE
            y_diff = (y_val_pred - y_val)
            perc_error = abs(100 * np.divide(y_diff, y_val, out=np.zeros_like(y_diff), where=y_val != 0))
            mape = perc_error.mean()

            # Max Percent Error
            max_perc_err = perc_error.max()

            self.journal[C.MAE] = mae
            self.journal[C.MSE] = mse
            self.journal[C.RMSE] = rmse
            self.journal[C.R2] = r2
            self.journal[C.NRMSE] = nrmse
            self.journal[C.MAX_ERR] = max_err
            self.journal[C.MAPE] = mape
            self.journal[C.MAX_PERC_ERR] = max_perc_err

            # For percent difference distribution
            pred_y = y_val_pred  # .flatten()
            num_points = X_val.shape[0]
            # model.evaluate(X_val, y_val, verbose=2)

            target_slice_i, target_slice_pd, target_perc_diff, target_y_truth = (
                Util.get_model_stats(y_val, pred_y))
            percent_errors = Util.calculate_percent_error(target_slice_pd)

            self.journal[C.PERCENT_ERRORS] = percent_errors

            #################### Plots ####################
            abs_error = abs(y_val_pred - y_val)
            text = '$R^2$ score: %.5f' % r2
            perc_text = '\nPercent Error:'
            perc_text = perc_text + Util.percent_error_text(percent_errors)

            self.custom_plots(j)
            self.actual_vs_predicted_plot(j, text, y_val, y_val_pred)
            self.percent_error_plot(j, perc_error, perc_text, y_val)
            self.absolute_error_plot(abs_error, j, y_val)

    def actual_vs_predicted_plot(self, j, text, y_val, y_val_pred):
        # Actual vs. Predicted Plot
        fig = plt.figure(figsize=(12, 8))
        plt.title(
            'Actual Value vs. Predicted Value of ' + self.state.target_variable + ' (Top Model #' + str(j) + ')')
        p1 = max(max(y_val), max(y_val_pred))
        p2 = min(min(y_val), min(y_val_pred))
        ideal = np.linspace(p1, p2, 10000)
        plt.plot(ideal, ideal, 'b-', label='Ideal')
        plt.plot(ideal, ideal * (1 + 0.2), 'g--', label='+/- 20% Error')
        plt.plot(ideal, ideal * (1 - 0.2), 'g--')
        plt.plot(ideal, ideal * (1 + 0.1), 'k:', label='+/- 10% Error')
        plt.plot(ideal, ideal * (1 - 0.1), 'k:')
        plt.plot(ideal, ideal * (1 + 0.05), 'c-.', label='+/- 5% Error')
        plt.plot(ideal, ideal * (1 - 0.05), 'c-.')
        plt.plot(ideal, ideal * (1 + 0.025), 'k--', label='+/- 2.5% Error')
        plt.plot(ideal, ideal * (1 - 0.025), 'k--')
        plt.scatter(y_val, y_val_pred, c='crimson')
        plt.xlabel('Actual Value of ' + self.state.target_variable, fontsize=18)
        plt.ylabel('Predicted Value of ' + self.state.target_variable, fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.text(1, 0, text, horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes,
                 fontsize=15, color='black')
        plt.grid(alpha=0.15)
        plt.legend()
        fig.savefig(
            self.state.plots_directory + 'actual-vs-predicted_' + self.state.model_name + '_top_' + str(j) + '.png')
        plt.close()

    def percent_error_plot(self, j, perc_error, perc_text, y_val):
        # Percent Error Plot - Log Scale
        fig = plt.figure(figsize=(12, 8))
        plt.title(
            'Percent Error between Actual Value and Predicted Value of ' + self.state.target_variable + ' (Top Model #'
            + str(j) + ')')
        plt.scatter(y_val, perc_error, c='crimson')
        plt.yscale('log')
        plt.xlabel('Actual Value of ' + self.state.target_variable, fontsize=15)
        plt.ylabel('Percent Error (%) of ' + self.state.target_variable, fontsize=15)
        plt.text(1, 1, perc_text, horizontalalignment='right', verticalalignment='top',
                 transform=plt.gca().transAxes, fontsize=15, color='black')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid(alpha=0.15)
        fig.savefig(
            self.state.plots_directory + 'percent-error_' + self.state.model_name + '_top_' + str(j) + '.png')
        plt.close()

    def absolute_error_plot(self, abs_error, j, y_val):
        # Absolute Error Plot
        fig = plt.figure(figsize=(12, 8))
        plt.title(
            'Absolute Error between Actual Value and Predicted Value of ' + self.state.target_variable +
            ' (Top Model #' + str(j) + ')')
        plt.scatter(y_val, abs_error, c='crimson')
        plt.xlabel('Actual Value of ' + self.state.target_variable, fontsize=15)
        plt.ylabel('Absolute Error of ' + self.state.target_variable, fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid(alpha=0.15)
        fig.savefig(
            self.state.plots_directory + 'absolute-error_' + self.state.model_name + '_top_' + str(j) + '.png')
        plt.close()
