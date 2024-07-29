import pickle
from dataclasses import dataclass, field

import matplotlib.pyplot as plt

from ampl.state import State
from ampl.eval import ModelEval
from ampl.constant import Constant as C
from ampl.convolutional_neural_network.util import Util as UtilCNN
from ampl.util import Util
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error
import numpy as np
from fastai.vision import *
from fastai import *
from fastai.vision.all import *

import logging

logger = logging.getLogger(__name__)

@dataclass
class PipelineModelEval(ModelEval):
    """
        evaluate models
        :param state:
        :type state: State
        :param callback: 
        :type state: object
    """
    state: State

    def __post_init__(self):
        super().__init__(C.EVALUATE_CNN, self.state, C.OPTUNA_CNN)

    def load_model(self, model_index: int, model_ext: str = C.MODEL_EXT_CNN):
        load_model_file = self.state.get_model_path(model_index, model_ext=model_ext)
        fa = Util.load_fastai
        learner = fa.Learner.load(load_model_file)
        logging.debug(learner.summary())

        return learner
    
    def custom_plots(self, model_index: int):
        file_base = self.state.get_model_base_path(model_index)
        load_history_file = file_base + C.HISTORY_EXT

        with open(load_history_file, "rb") as history_file:
            r = pickle.load(history_file)

        self.loss_plot(model_index, r['loss'], r['val_loss'])

    def loss_plot(self, j, loss, val_loss):
        fig = plt.figure(figsize=(20, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        fig.subtitle('Training and Validation Loss Comparison')
        ax1.plot(loss, label='loss')
        ax1.plot(val_loss, label='val_loss')
        ax1.set(xlabel='Epoch', ylabel='Loss - MAE')
        ax1.legend()

        thresh = 0.04
        m = [x for x in loss if x <= thresh]
        n = [x for x in val_loss if x <= thresh]
        ax2.plot(m, label='loss')
        ax2.plot(n, label='val_loss')
        ax2.set(xlabel='Epoch', ylabel='Loss - MAE')
        ax2.legend()
        fig.savefig(
            self.state.plots_directory + 'loss_' + self.state.model_name + '_top_' + str(j) + '.png')
        plt.close()


    def run(self, random_state: int = 0, model_ext=C.MODEL_EXT_CNN):
        logging.debug('Running CNN Eval step')
        Util.check_and_create_directory(self.state.results_directory)
        Util.check_and_create_directory(self.state.saved_models_directory)
        Util.check_and_create_directory(self.state.plots_directory)

        dataloader = self.state.data

        # input_shape = (dataloader.train_size, )
        # logging.debug(f'{input_shape = }')


        # y_train = self.data.denormalize_y(y_train)
        # y_test = self.data.denormalize_y(y_test)
        # y_val = self.data.denormalize_y(y_val)

        logging.debug(f'{self.state.num_models = }')
        self.journal[C.N_MODELS] = self.state.num_models

        fa = UtilCNN.load_fastai()
        lrs = slice(self.lr/400, self.lr/4)

        for j in range(self.state.num_models):
            results = dict()
            self.journal[f'model_{j}'] = results

            load_model_file = self.state.get_model_path(j, model_ext=model_ext)
            learner = fa.load_learner(load_model_file)

            # Make Predictions
            # y_train_pred = model.predict(X_train)
            # y_test_pred = model.predict(X_test)
            # y_val_pred = model.predict(X_val)
            pred_y = learner.fit_flat_cos(self.epochs, lrs, cbs=GradientAccumulation(n_acc=8))

            pred_y = np.reshape(pred_y, (-1, 1))
            # y_train_pred = np.reshape(y_train_pred, (-1, 1))
            # y_test_pred = np.reshape(y_test_pred, (-1, 1))
            # y_val_pred = np.reshape(y_val_pred, (-1, 1))

            pred_y  = self.data.denormalize_y(pred_y)
            # y_train_pred = self.data.denormalize_y(y_train_pred)
            # y_test_pred = self.data.denormalize_y(y_test_pred)
            # y_val_pred = self.data.denormalize_y(y_val_pred)

            # Mean Absolute Error
            mae = mean_absolute_error(y_val, pred_y)

            # Mean Square Error
            mse = mean_squared_error(y_val, pred_y)

            # Root Mean Square Error
            rmse = np.sqrt(mean_squared_error(y_val, pred_y))

            # R²
            r2 = r2_score(y_val, pred_y)  # Best possible score is 1.0, lower values are worse

            # Normalized Root Mean Squared Error
            nrmse = rmse / (y_val.max() - y_val.min())

            # Max Error
            max_err = max_error(y_val, pred_y)

            # MAPE
            y_diff = (pred_y - y_val)
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
            # pred_y = y_val_pred  # .flatten()
            # num_points = X_val.shape[0]
            # model.evaluate(X_val, y_val, verbose=2)

            target_slice_i, target_slice_pd, target_perc_diff, target_y_truth = (
                Util.get_model_stats(y_val, pred_y))
            percent_errors = Util.calculate_percent_error(target_slice_pd)

            self.journal[C.PERCENT_ERRORS] = percent_errors

            #################### Plots ####################
            abs_error = abs(pred_y - y_val)
            text = '$R^2$ score: %.5f' % r2
            perc_text = '\nPercent Error:'
            perc_text = perc_text + Util.percent_error_text(percent_errors)

            self.custom_plots(j)
            self.actual_vs_predicted_plot(j, text, y_val, pred_y)
            self.percent_error_plot(j, perc_error, perc_text, y_val)
            self.absolute_error_plot(abs_error, j, y_val)


