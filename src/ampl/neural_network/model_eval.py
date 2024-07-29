import pickle
from dataclasses import dataclass, field

import matplotlib.pyplot as plt

from ampl.state import State
from ampl.eval import ModelEval
from ampl.constant import Constant as C
from ampl.neural_network.util import Util


import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineModelEval(ModelEval):
    """
        Create the ensemble of good models to perform predictions with. This code requires models to be in
        the database to perform different ensemble methods.
        :param state:
        :type state: State
        :param callback:
        :type state: object
    """
    state: State

    def __post_init__(self):
        super().__init__(C.EVALUATE_NN, self.state, C.NN)

    def load_model(self, model_index: int, model_ext: str = C.MODEL_EXT_NN):
        load_model_file = self.state.get_model_path(model_index, model_ext=model_ext)
        tf = Util.load_tensorflow()
        model = tf.keras.models.load_model(load_model_file)
        logging.debug(model.summary())

        return model

    def custom_plots(self, model_index: int):
        file_base = self.state.get_model_base_path(model_index)
        load_history_file = file_base + C.HISTORY_EXT

        with open(load_history_file, "rb") as history_file:
            r = pickle.load(history_file)

        self.loss_plot(model_index,  r['loss'], r['val_loss'])

    def loss_plot(self, j, loss, val_loss):
        # Loss Plot
        fig = plt.figure(figsize=(20, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        fig.suptitle('Training and Validation Loss Comparison')
        ax1.plot(loss, label='loss')
        ax1.plot(val_loss, label='val_loss')
        ax1.set(xlabel='Epoch', ylabel='Loss - MAE')
        ax1.legend()
        # drop everything above threshold for a better look
        thresh = 0.04
        m = [x for x in loss if x <= thresh]
        n = [x for x in val_loss if x <= thresh]
        ax2.plot(m, label='loss')
        ax2.plot(n, label='val_loss')
        ax2.set(xlabel='Epoch', ylabel='Loss - MAE')
        ax2.legend()
        fig.savefig(
            f'{self.state.plots_directory}{self.key}_loss_{self.state.model_name}_top_{j}.png')
        plt.close()
