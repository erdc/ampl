
from dataclasses import dataclass, field
import itertools
# import the file with the user input
import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score, max_error

from ampl.neural_network.util import Util as UtilNN

from ampl.state import State
from ampl.pipelinestep import PipelineStep
from ampl.constant import Constant as C
from ampl.util import Util
from ampl.enums import EnsembleMode
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineModelEnsemble(PipelineStep):
    """
        Create the ensemble of good models to perform predictions with. This code requires models to be in
        the database to perform different ensemble methods.
        :param ensemble_mode:
        :param ensemble_results_file:
        :type state: object
    """
    state: State
    ensemble_mode: EnsembleMode
    num_models: int
    num_points: int = field(default=None, init=False)

    def __post_init__(self):
        super().__init__(C.ENSEMBLE_NN, self.state, C.NN)
        self._ensemble_mode = self.ensemble_mode.value

    @staticmethod
    def weights_search(models, X_val, y_val):
        """ Cartesian product grid search to choose weights """
        w = [i * 0.1 for i in range(11)]
        best_score, best_weights = 1.0, None

        for weights in itertools.product(w, repeat=len(models)):
            # skip if all weights are the same
            if len(set(weights)) == 1:
                continue

            # normalize weights and evaluate weighted model
            weights = Util.normalize_weights(weights)
            pred_y_wgt = Util.weighted_ens_pred(models, weights, X_val)
            score = mean_absolute_percentage_error(y_val, pred_y_wgt)
            if score < best_score:
                best_score, best_weights = score, weights
                print('>%s %.3f' % (best_weights, best_score))

        return list(best_weights)

    #################################################################

    def run(self, random_state: int = 0, model_ext=C.MODEL_EXT_NN):
        t0 = time.time()

        Util.check_and_create_directory(self.state.results_directory)
        Util.check_and_create_directory(self.state.plots_directory)
        Util.check_and_create_directory(self.state.saved_models_directory)

        X_train, X_val, X_test, y_train, y_val, y_test = self.state.data.train_val_test_split(random_state=random_state)

        self.journal[C.TRAINING_POINTS] = X_train.shape[0]
        self.journal[C.TESTING_POINTS] = X_test.shape[0]
        self.journal[C.VALIDATION_POINTS] = X_val.shape[0]

        input_shape = (X_train.shape[1], )
        logging.debug(f'{input_shape = }')

        t4 = time.time()
        logging.debug('>>> Loading/Training models...')

        nn_models = []
        self.journal[f'num_models'] = self.num_models
        # TODO : Need some logic to handle when ensemble step run as a separate step with more num of models
        #  than it was originally ran, Ensemble expects models to be in saved models folder, as it doesn't build its own
        #  so when it is not there it will error out

        tf = UtilNN.load_tensorflow()

        for j in range(self.num_models):

            load_model_file = self.state.get_model_path(j, model_ext=model_ext)
            mod = tf.keras.models.load_model(load_model_file)
            nn_models.append(mod)
            logging.debug(mod)

        #         # for i in np.arange(3,5):
        #         for i in np.arange(self.model_ensemble_pipeline.NUM_MODLES):
        #         #     if i != 1:
        #                 # TODO: figure out how to save models that will be ensembled
        #                 mod = keras.models.load_model(f'{self.model_ensemble_pipeline.SAVED_MODELS_DIRECTORY}top{i+1}.h5')
        #                 nn_models.append(mod)
        #                 print(mod)

        pred_y = []
        if self._ensemble_mode == EnsembleMode.NN_SIMPLE_AVG.value:  # simple average
            pred_y = [list(model.predict(X_val)) for model in nn_models]
            pred_y = np.mean(np.array(pred_y), axis=0)

        elif self._ensemble_mode == EnsembleMode.NN_MEDIAN.value:  # median
            pred_y = [list(model.predict(X_val)) for model in nn_models]
            pred_y = np.median(np.array(pred_y), axis=0)

        # TODO: this ensemble method does not work currently. WE NEED TO FIX
        elif self._ensemble_mode == EnsembleMode.NN_WEIGHTED_AVG.value:  # weighted average
            best_weights = self.weights_search(nn_models, X_val, y_val)
            pred_y = Util.weighted_ens_pred(nn_models, best_weights, X_val)

        t_train = time.time() - t4
        print(f'    Done: model training took {Util.display_time(t_train)}')

        # Denormalize for evaluation

        y_val_pred = self.data.denormalize_y(pred_y)
        y_val = self.data.denormalize_y(y_val)

        #################### State Evaluation ####################
        t5 = time.time()
        print('>>> Evaluating models...')

        ## Regression metrics

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

        # Max Percent Error
        actual = y_val#.to_numpy().reshape(-1, 1)
        y_diff = (y_val_pred - actual)
        percent_error = abs(100 * np.divide(y_diff, actual, out=np.zeros_like(y_diff), where=actual != 0))
        mape = percent_error.mean()
        max_perc_err = percent_error.max()

        self.journal[C.MAE] = mae
        self.journal[C.MSE] = mse
        self.journal[C.RMSE] = rmse
        self.journal[C.R2] = r2
        self.journal[C.NRMSE] = nrmse
        self.journal[C.MAX_ERR] = max_err
        self.journal[C.MAPE] = mape
        self.journal[C.MAX_PERC_ERR] = max_perc_err

        # For percent difference distribution

        pred_y_flat = y_val_pred#.flatten()

        target_slice_i, target_slice_pd, target_perc_diff, target_y_truth = (
            Util.get_model_stats(y_val, pred_y_flat))
        percent_errors = Util.calculate_percent_error(target_slice_pd)

        self.journal[C.PERCENT_ERRORS] = percent_errors

        print(f'    Done: evaluation took {Util.display_time(time.time() - t5)}')

        #################### Model Visualizations ####################
        t6 = time.time()
        print('>>> Plotting...')

        abs_error = abs(y_val_pred - actual)
        text = '$R^2$ score: %.5f' % r2
        perc_text = '\nPercent Error:'
        perc_text += Util.percent_error_text(percent_errors)

        # Actual vs. Predicted
        fig = plt.figure(figsize=(12, 8))
        plt.title('Actual Value vs. Predicted Value of ' + self.state.target_variable)
        p1 = max(max(y_val), max(y_val_pred))
        p2 = min(min(y_val), min(y_val_pred))
        ideal = np.linspace(p1, p2, 10000)
        # plt.plot([p1, p2], [p1, p2], 'b-')
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
        plt.xlabel('Actual Value of ' + self.state.target_variable, fontsize=15)
        plt.ylabel('Predicted Value of ' + self.state.target_variable, fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.text(1, 0, text, horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes,
                 fontsize=15, color='black')
        plt.grid(alpha=0.15)
        plt.legend()
        fig.savefig(
            self.state.plots_directory + 'ensemble_' + self._ensemble_mode + '_actual-vs-predicted_' +
            self.state.model_name + '.png')
        plt.close()

        # Percent Error - Log Scale
        fig = plt.figure(figsize=(12, 8))
        plt.title(
            'Percent Error between Actual Value and Predicted Value of ' + self.state.target_variable)
        plt.scatter(y_val, percent_error, c='crimson')
        plt.yscale('log')
        # plt.ylim((1e-6, 1e2))
        plt.xlabel('Actual Value of ' + self.state.target_variable, fontsize=15)
        plt.ylabel('Percent Error (%) of ' + self.state.target_variable, fontsize=15)
        plt.text(1, 1, perc_text, horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
                 fontsize=15, color='black')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid(alpha=0.15)
        fig.savefig(
            self.state.plots_directory + 'ensemble_' + self._ensemble_mode + '_percent-error_' +
            self.state.model_name + '.png')
        plt.close()

        # Absolute Error
        fig = plt.figure(figsize=(12, 8))
        plt.scatter(y_val, abs_error, c='crimson')
        plt.xlabel('Actual Value of ' + self.state.target_variable, fontsize=15)
        plt.ylabel('Absolute Error of ' + self.state.target_variable, fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid(alpha=0.15)
        fig.savefig(
            self.state.plots_directory + 'ensemble_' + self._ensemble_mode + '_absolute-error_' +
            self.state.model_name + '.png')
        plt.close()

        self.journal[C.RUN_TIME] = Util.display_time(t6 - t0)

    @property
    def base_filename(self):
        return super().base_filename + '_' + self._ensemble_mode
