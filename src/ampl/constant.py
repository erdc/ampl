from typing import Final, Tuple


class Constant(object):
    NN : Final[str] = 'NN'
    DT : Final[str] = 'DT'
    CNN : Final[str] = 'CNN'


    TARGET_VARIABLE: Final[str] = 'target_variable'
    # Dense fully connected neural network
    BUILD_NN: Final[str] = 'build_nn'
    OPTUNA_NN: Final[str] = 'optuna_nn'
    EVALUATE_NN: Final[str] = 'eval_nn'
    ENSEMBLE_NN: Final[str] = 'ensemble_nn'

    # Decision Trees 
    BUILD_DT: Final[str] = 'build_dt'
    OPTUNA_DT: Final[str] = 'optuna_dt'
    EVALUATE_DT: Final[str] = 'eval_dt'
    ENSEMBLE_DT: Final[str] = 'ensemble_dt'

    BUILD_CNN: Final[str] = 'build_cnn'
    OPTUNA_CNN: Final[str] = 'optuna_cnn'
    EVALUATE_CNN: Final[str] = 'eval_cnn'
    ENSEMBLE_CNN: Final[str] = 'ensemble_cnn'

    TOTAL_POINTS: Final[str] = 'total_points'
    TRAINING_POINTS: Final[str] = 'training_points'
    TESTING_POINTS: Final[str] = 'testing_points'
    VALIDATION_POINTS: Final[str] = 'validation_points'

    MODEL: Final[str] = 'model'

    MODEL_NAME: Final[str] = 'model_name'
    STUDY_NAME: Final[str] = 'study_name'
    N_MODELS: Final[str] = 'num_models'
    N_FEATURES: Final[str] = 'n_features'

    RESULTS_DIR: Final[str] = 'results_directory'
    SAVED_MODELS_DIR: Final[str] = 'saved_models_directory'
    PLOTS_DIR: Final[str] = 'plots_directory'
    DATA: Final[str] = 'data'
    FEATURE_IMPORTANCE: Final[str] = 'feature_importance'
    RESULTS: Final[str] = 'results'
    TRAIN_FRAC: Final[str] = 'train_frac'
    TEST_FRAC: Final[str] = 'test_frac'
    METADATA_FILE: Final[str] = 'metadata_file'
    METADATA_JSON: Final[str] = 'metadata.json'

    N_TRIALS: Final[str] = 'n_trials'
    N_PRUNED_TRIALS: Final[str] = 'n_pruned_trials'
    N_FAILED_TRIALS: Final[str] = 'n_failed_trials'
    N_COMPLETED_TRIALS: Final[str] = 'n_completed_trials'
    BEST_TRIAL_RUN: Final[str] = 'best_trial_run'
    BEST_TRIAL: Final[str] = 'best_trial'
    TOP_TRIALS: Final[str] = 'top_trials'
    EPOCHS_SET: Final[str] = 'epochs_set'
    N_RUNS: Final[str] = 'n_runs'
    RUN_TIME: Final[str] = 'run_time_sec'
    RUN_EPOCHS: Final[str] = 'run_epochs'

    MAE: Final[str] = 'mae'
    MSE: Final[str] = 'mse'
    RMSE: Final[str] = 'rmse'
    R2: Final[str] = 'r2'
    NRMSE: Final[str] = 'nrmse'
    MAX_ERR: Final[str] = 'max_err'
    MAPE: Final[str] = 'mape'
    MAX_PERC_ERR: Final[str] = 'max_perc_err'

    PERCENT_ERRORS: Final[str] = 'percent_errors'
    DT_PARAMS: Final[str] = 'params'

    # Extension formats for neural networks, decision trees, and convolutional neural networks
    MODEL_EXT_NN: Final[str] = '.keras'
    MODEL_EXT_DT: Final[str] = '.json'
    MODEL_EXT_CNN: Final[str] = '.pt'
    
    CSV_EXT: Final[str] = '.csv'
    HISTORY_EXT: Final[str] = '_history.pickle'

    ERROR_PERCENTAGES: Final[Tuple[int]] = (20, 10, 5, 2.5, 1, 0.5)

    LEARNING_RATE: Final[str] = 'learning_rate'
    OPTIMIZER: Final[str] = 'optimizer'
    ACTIVATION: Final[str] = 'activation'
    N_LAYERS: Final[str] = 'n_layers'
    N_UNITS_LAYER: Final[str] = 'n_units_layer_'
    N_UNITS_INPUT_LAYER: Final[str] ='n_units_inputl'

    VERBOSITY: Final[str] = 'verbosity'
    OBJECTIVE : Final[str] = 'objective'
    N_ESTIMATORS : Final[str] = 'n_estimators'
    MAX_DEPTH : Final[str] = 'max_depth'
    SUBSAMPLE : Final[str] = 'subsample'
    ALPHA : Final[str] = 'alpha'
    LAMBDA : Final[str] = 'lambda'
    GAMMA : Final[str] = 'gamma'
    SEED : Final[str] = 'seed'
    COLSAMPLEBYTREE : Final[str] = 'colsample_bytree'
    MINCHILDWEIGHT : Final[str] = "min_child_weight"



    LOSS: Final[str] = 'loss'
    VAL_LOSS: Final[str] = "val_loss"

    OPTUNA_MONITOR: Final[str] = VAL_LOSS
    EARLY_STOPPING_MONITOR: Final[str] = LOSS
    EARLY_STOPPING_PATIENCE: Final[int] = 10
