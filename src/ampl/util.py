# os.py
import functools
import json
import os
import logging
from pathlib import Path
import shutil

import importlib_resources
import numpy as np
import random
from ampl.constant import Constant as C


class Util(object):
    @staticmethod
    def check_and_create_directory(directory_path: str):
        """
        Checks to see if the 'results', 'results/saved_models', 'results/plots', etc. directories exist.
        If the directory does not exist, this function will create the directory.

        :param directory_path: This is the path to the directory that needs to be checked.
        :type directory_path: str
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    @staticmethod
    def add_slash(path):
        """
        Adds a slash to the path, but only when it does not already have a slash at the end
        Params: a string
        Returns: a string
        """
        if not path.endswith('/'):
            path += '/'
        return path

    @staticmethod
    def get_n_jobs(percent=0.9):
        n_jobs = -1  # Let Optuna use all CPU
        n_cpu = os.cpu_count()

        if n_cpu > 2 and percent:
            if 0.1 < percent <= .9:
                n_jobs = int(n_cpu * percent)

        return n_jobs

    @staticmethod
    def create_default_config_file(config_file):
        if Path.is_file(Path(config_file)):
            logging.warning(
                'Overwrite Warning: '
                'There is an existing configuration file with the same name!! '
                'File will be overwritten!')
        my_resources = importlib_resources.files(__package__)
        shutil.copyfile(str(my_resources / 'default_config.yml'), config_file)

    @staticmethod
    def update_journal(key: str, journal: dict, filename: str):
        """
        Persists the journal dictionary as a
        :param filename:
        :param key: unique key to persist the journal dictionary
        :type key: str
        :param journal: metrics, logs and pertinent data to persist
        :type journal: dict
        :return:
        """

        full_journal = dict()
        if os.path.exists(filename):
            with open(filename, 'r') as fp:
                full_journal = json.load(fp)
        # if key in full_journal:
        #     full_journal[key].update(journal)
        # else:
        full_journal[key] = journal

        with open(filename, 'w') as fp:
            json.dump(full_journal, fp, indent=2)

    @staticmethod
    def persist_journal(func, name: str, journal: dict, filename: str):
        """
        :param func:
        :param name:
        :param journal:
        :param filename:
        :return:
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logging.info(f'Running {name}')
            value = func(*args, **kwargs)
            Util.update_journal(name, journal, filename)
            return value

        return wrapper

    @staticmethod
    def get_model_stats(val_y, pred_y):
        """
        Evaluates a neural network on the validation set.

        :param val_y: Validation set target.
        :param pred_y: Model predicted target

        :return: slice_i, slice_pd, percent difference results on the df, and model_y_truth
        :rtype: _type_
        """
        num_points = val_y.shape[0]

        # renaming arrays to numpy arrays with new variable names
        model_y_pred, model_y_truth = pred_y, val_y
        model_y_diff = model_y_pred - model_y_truth
        model_y = np.divide(model_y_diff, model_y_truth, out=np.zeros_like(model_y_diff), where=model_y_truth != 0)
        # obtaining a random slice of df, where _i are the indices and _pd are the perc diff values
        # perc_diff = np.abs(100 * (model_y_pred - model_y_truth) / model_y_truth)
        perc_diff = np.abs(100 * model_y)
        slice_i = random.sample(range(len(model_y_truth)), num_points)  # choose the num_point points to plot at the end
        slice_pd = perc_diff[slice_i]
        # slice_pd = [perc_diff[i] for i in slice_i]

        return slice_i, slice_pd, perc_diff, model_y_truth

    @staticmethod
    def calculate_percent_error(perc_error, percentages=C.ERROR_PERCENTAGES):
        """

        :param perc_error:
        :param percentages:
        :return:
        """
        num_points = len(perc_error)

        perc_error = np.reshape(perc_error, (-1, 1))

        summary = dict()
        for perc in percentages:
            summary[perc] = (len([pt for pt in perc_error if pt < perc]) * 100 / num_points)

        return summary

    @staticmethod
    def percent_error_text(perc_errors):
        """

        :param perc_errors:
        :return:
        """
        perc_text = ''
        for k, v in perc_errors.items():
            perc_text += f'\n  Under {k}%:  {v : .1f}%'
        return perc_text

    @staticmethod
    def display_time(time_in_seconds):
        """
        Display time in human-readable units.

        :param time_in_seconds:
        :return:
        """
        unit_labels = ['sec', 'min', 'hrs']
        unit_id = 0
        t = time_in_seconds
        while t > 60 and unit_id <= 2:
            t /= 60
            unit_id += 1
        return f'{t:.4f} {unit_labels[unit_id]}'

    # Weighted Average Selection
    @staticmethod
    def weighted_ens_pred(models, weights, test_x_normed):
        """
        Apply best weights from grid search to model predictions
        :param models:
        :param weights:
        :param test_x_normed:
        :return:
        """
        pred_y = [list(model.predict(test_x_normed)) for model in models]
        pred_y = np.array(pred_y)
        pred_y_wgt = np.tensordot(pred_y, weights, axes=((0), (0)))
        return pred_y_wgt

    @staticmethod
    def normalize_weights(weights):
        """
        Scales weights to add up a unit norm vector (calculating L1 norm and dividing each weight by that value)
        :param weights:
        :return:
        """
        results = np.linalg.norm(weights, 1)
        return weights if results == 0.0 else (weights / results)

    @staticmethod
    def load_optuna():
        """
        Lazy loading optuna
        :return: optuna module
        """
        import optuna
        return optuna
    
    @staticmethod
    def relativePathHelper(folderPath):
        """
        Helper function to locate the relative path to a file. Print the files located at the provided relative path
        :return: relative path
        """
        # List all files in the folder
        files = os.listdir(folderPath)

        # Loop through the files and print their contents
        for file_name in files:
            file_path = os.path.join(folderPath, file_name)
            print(file_path)

    @staticmethod
    def write_dict_to_json(dict_, json_file):
        """
        Saves a dictionary to a .json file
        """
        json_data = json.dumps(dict_, indent=4)
        with open(json_file, "w") as outfile:
            outfile.write(json_data)
