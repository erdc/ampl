import os
from dataclasses import dataclass, field
from typing import Tuple, List, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance():
    """
        Feature Importance class studies Shapley Additive Explanations (SHAP) importance based on XGBoost Regressor
    """

    number_of_features: int = 9
    """Top n features to use"""

    feature_list: List[str] = None
    """List of features to include in study"""

    test_size: float = 0.2
    """Percentage of test size to use in study, default is 0.2 (20%)"""

    name: str = field(default='feature_importance', init=False)
    """Name of feature importance study, must be without any whitespace, used in path name"""

    results_df: pd.DataFrame = field(default=None, init=False, repr=False, compare=False)
    """Resulting important features and values from SHAP study"""

    shap_values: np.ndarray = field(default=None, init=False, repr=False, compare=False)
    """Resulting values from SHAP TreeExplainer study"""

    @property
    def feature_importance_list(self):
        """A list of top number of features from study """
        if self.results_df is not None:
            return self.results_df['feature'].to_list()[:self.number_of_features]
        else:
            return list()

    def run(self, df_X: pd.DataFrame, df_y: pd.Series, random_state: int = 0) -> list[str]:
        """Executes feature importance study using SHAP TreeExplainer based on XGBoost model"""
        import shap
        import xgboost

        if self.feature_list:
            df_X = df_X[self.feature_list]

        y = df_y

        X_train, X_test, y_train, y_test = train_test_split(
            df_X, y, test_size=self.test_size, random_state=random_state)

        xgb = xgboost.XGBRegressor()
        xgb.fit(X_train, y_train)

        explainer = shap.TreeExplainer(xgb)
        self.shap_values = explainer.shap_values(X_test)

        vals = np.abs(self.shap_values).mean(0)
        self.results_df = pd.DataFrame(list(zip(df_X.columns, vals)),
                                       columns=['feature', 'importance_values'])
        self.results_df.sort_values(by=['importance_values'], ascending=False, inplace=True)

        return self.feature_importance_list
