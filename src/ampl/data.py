# df.py
from abc import abstractmethod
from dataclasses import dataclass, field
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Union, Optional, Literal, List, Any, Dict
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

import pandas as pd
from sklearn.model_selection import train_test_split

from ampl.feature_importance import FeatureImportance
import logging

logger = logging.getLogger(__name__)


@dataclass
class Database:
    """
    Database class to hold sqllite3 object
    """
    data_file: str
    """SQLite3 data file path"""

    def connection(self) -> sqlite3.Connection:
        """
        sqllite3 connection context manager that closes connection automatically
        :rtype: sqlite3.Connection
        """
        return self.connect(self.data_file)

    @staticmethod
    @contextmanager
    def connect(sql_data_file: str) -> sqlite3.Connection:
        """
        sqllite3 connection context manager that closes connection automatically
        :rtype: sqlite3.Connection
        :param sql_data_file:
        """
        conn = sqlite3.connect(sql_data_file)
        try:
            yield conn
        except Exception as e:
            # do something with exception
            conn.rollback()
            raise e
        else:
            conn.commit()
        finally:
            conn.close()


class IData:
    """
    Data Interface
    """

    @abstractmethod  # The innermost decorator
    def train_test_split(self, *args, **kwargs) -> object:
        raise NotImplementedError

    @abstractmethod  # The innermost decorator
    def train_val_test_split(self, *args, **kwargs) -> object:
        raise NotImplementedError


@dataclass
class Data(IData):
    """
    Data class encapsulates all the input data related functionalities, functions such as IO operations,
    normalization, standardization, and feature importance study
    """
    df: pd.DataFrame = field(compare=False, repr=False)
    """ Main dataframe with the features (X) and target variable (y) in Pandas dataframe format """

    target_variable: str
    """ target variable (y) column name"""

    feature_list: List[str] = None
    """List of features to include in study"""

    feature_importance: FeatureImportance = field(default_factory=lambda: FeatureImportance(), repr=False)
    """ Feature Importance to run feature importance study using AutoML(XGBoost)  """

    cols_to_enum: List[str] = field(default_factory=list)
    """ Columns to enumerate, takes the identified categorical columns and converts it to ordinal(numerical)"""

    target_col_function: Callable = field(default=None, compare=False, repr=False)
    """ A callback function to apply or calculate target variable (y)"""

    train_size: float = 0.8
    """ Percentage of data to use for training, default set to 80%, that is .8"""

    val_size: float = 0.1
    """ Percentage of data to use for validation, default set to 10%, that is .1, whatever percentage left is used for test data"""

    test_size: float = 0.1
    """ Percentage of data to use for training, default set to 10%, that is .1 """

    feature_importance_list: List[str] = field(default=None, init=False)
    """ A list of features to consider for feature importance study """

    encoders: Dict = field(default_factory=dict, init=False, compare=False)
    """ Encoder dictionary to encode/decode categorical data"""

    encoder_mapping: Dict = field(default_factory=dict, init=False, compare=False)
    """ Dictionary to save mapping for categorical data"""

    column_stats: Dict = field(default_factory=dict, init=False, compare=False)
    """ Dictionary to save stats used for normalizing"""

    def __post_init__(self):
        if not isinstance(self.df, pd.DataFrame):
            raise ValueError('Field "df" must be of Pandas Dataframe type and cannot be None')

        if self.feature_list:
            feature_list_ = set(self.feature_list)
            if self.target_variable and self.target_col_function is None:
                feature_list_.add(self.target_variable)
            self.df = self.df[list(feature_list_)]

        self._verify_target_variable()
        self._transform_data()

        assert 1.0 == (self.train_size + self.val_size + self.test_size), \
            'Training, test, and validation size (percentages) does not add up to 1 (100 %)'

    @property
    def df_X(self) -> pd.DataFrame:
        """
        Dataframe Property containing only the top n features from feature importance study
        """
        _df_X = self.df.drop(self.target_variable, axis=1, errors='ignore')
        if self.feature_importance_list:
            _df_X = _df_X.filter(self.feature_importance_list)
        return _df_X

    @property
    def df_y(self) -> pd.DataFrame:
        """
            Target Property as pd.DataFrame
        :return:
        """
        return self.df.filter([self.target_variable])

    def train_test_split(self, train_size: float = None, random_state: int = 0,
                         stratify: Any = None) -> object:
        """
        Splits the given data into training, and test data. Training dataset size is the complement of
        test size.
        Training and Test data are used to build model
        :param stratify: bool
        :param train_size: float, default is set to .8 (80%)
        :param random_state: int, default 0
        :return: a tuple of  X_train, X_test, y_train, y_test
        """
        y = self.df_y.squeeze().values.reshape(-1, 1)
        X = self.df_X.values

        if train_size is None:
            train_size = self.train_size

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size,
                                                            random_state=random_state, stratify=stratify)

        return X_train, X_test, y_train, y_test

    def train_val_test_split(self, val_size: float = None, test_size: float = None, random_state: int = 0,
                             stratify: Any = None) -> object:
        """
        Splits the given data into training, validation and test data. Training dataset size is the complement of
        validation and test size.
        Training and Test data are used to build model, and Validation data is reserved as unseen data to evaluate the model
        :param stratify: bool
        :param val_size: float, default is set to .1 (10%)
        :param test_size: float, default is set to .1 (10%)
        :param random_state: int, default 0
        :return: a tuple of  X_train, X_val, X_test, y_train, y_val, y_test
        """

        if test_size is None:
            test_size = self.test_size
        if val_size is None:
            val_size = self.val_size

        X_train, x_val_test, y_train, y_val_test = self.train_test_split(train_size=(1 - (val_size + test_size)),
                                                                         random_state=random_state, stratify=stratify)
        relative_test_size = test_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=relative_test_size,
                                                        random_state=random_state, stratify=stratify)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _transform_data(self):
        # enumerate and columns that are strings
        self.enum_columns()

        self._process_target_col()

        # Normalize df between 0 and 1
        self.normalize()

        if self.feature_importance:
            self.feature_importance_list = self.feature_importance.run(self.df_X, self.df_y.squeeze())

    def _process_target_col(self):
        # calculate target column, if necessary. Otherwise convert target column to the 'float' datatype
        if self.target_col_function:
            self.df[self.target_variable] = self.target_col_function(self.df_X)
        if self.df_y is not None:
            self.df[self.target_variable] = self.df_y.astype('float')

            self.target_stats = self.df_y[self.target_variable].describe()

    def _verify_target_variable(self):
        if (self.target_col_function is None and
                (self.df_y is None or
                 self.target_variable is None or
                 self.target_variable not in self.df)):
            raise AttributeError(
                'Target Variable is missing in dataframe, and Target Variable Function is set to None,' +
                ' at least one of them is required!')

    def enum_columns(self):
        """
        Enumerates any columns that contain string values.
        Accepts a dataframe and list of columns that need to be converted from string to enum.
        The return is a dataframe with the columns converted.
        """
        if self.cols_to_enum:
            for col in self.cols_to_enum:
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(self.df[[col]])
                category = self.encoders[col].classes_
                # self.encoder_mapping[col] = dict(zip(category, self.encoders[col].transform(category)))
                self.encoder_mapping[col] = dict(zip(category, list(map(int, self.encoders[col].transform(category)))))
                self.df[col] = self.encoders[col].transform(self.df[[col]])

    def decode_enum(self, df_):
        """
        Coverts data that was enumerated back to categorical data
        Accepts a dataframe and list of columns that need to be converted from numerical to str.
        The return is a dataframe with the columns converted.
        """
        if self.cols_to_enum is not None:
            for col in self.cols_to_enum:
                if col in df_.columns:  # checking if enum column is actually in the final dataframe after feature importance
                    df_[col] = pd.DataFrame(self.encoders[col].inverse_transform(df_[[col]].astype(int)), columns=[col])

        return df_

    def normalize(self):
        """
        Normalizes the df between 0 and 1.
        """
        self.stats = self.df.describe()
        if len(self.stats.columns) != len(self.df.columns):
            raise ValueError("Dataframe includes string data which was not enumerated")
        for col in self.df:
            self.column_stats[col] = {'min': self.stats[col]['min'], 'max': self.stats[col]['max']}
            self.df[col] = self.df[col].apply(
                lambda x: (x - self.stats[col]['min']) / (self.stats[col]['max'] - self.stats[col]['min']))

    # Denormalize for evaluation
    def denormalize_y(self, y):
        """
        Denormalizes the neural network's predictions for model evaluation.

        :param y: The dataset that needs to be denormalized.
        :type y: pandas Series or NumPy array
        :return: A denormalized dataset
        :rtype: pandas Series or NumPy array
        """
        return y * (self.target_stats['max'] - self.target_stats['min']) + self.target_stats['min']

    def denormalize(self, df_):
        """
        Denormalizes the input for printing predictions.
        """
        for col in df_:
            df_[col] = df_[col].apply(
                lambda x: x * (self.stats[col]['max'] - self.stats[col]['min']) + self.stats[col]['min'])

        df_ = self.decode_enum(df_)

        return df_


class PreSplitData(Data):
    def __init__(self, df_train, df_test, df_val,
                 target_variable: str,
                 feature_importance: FeatureImportance = None,
                 feature_list: List[str] = None,
                 cols_to_enum: List[str] = None,
                 target_col_function: Callable = None
                 ):

        if feature_list:
            feature_list_ = set(feature_list)
        else:
            feature_list_ = set(df_train.columns.tolist())

        if target_variable and target_col_function is None:
            feature_list_.add(target_variable)
        feature_list_ = list(feature_list_)

        self.df_train = df_train[feature_list_]
        self.df_test = df_test[feature_list_]
        self.df_val = df_val[feature_list_]

        super().__init__(self.df, target_variable, feature_importance=feature_importance, feature_list=feature_list,
                         cols_to_enum=cols_to_enum, target_col_function=target_col_function)

    @property
    def df(self):
        return pd.concat([self.df_train, self.df_val, self.df_test], axis=0, ignore_index=True)

    @df.setter
    def df(self, value):
        pass

    @property
    def df_train_X(self) -> pd.DataFrame:
        """
        Dataframe Property containing only the top n features from feature importance study
        """
        _df_X = self.df_train.drop(self.target_variable, axis=1, errors='ignore')
        if self.feature_importance_list:
            _df_X = _df_X.filter(self.feature_importance_list)
        return _df_X

    @property
    def df_train_y(self) -> pd.DataFrame:
        """
            Target Property as pd.DataFrame
        :return:
        """
        return self.df_train.filter([self.target_variable])

    @property
    def df_test_X(self) -> pd.DataFrame:
        """
        Dataframe Property containing only the top n features from feature importance study
        """
        _df_X = self.df_test.drop(self.target_variable, axis=1, errors='ignore')
        if self.feature_importance_list:
            _df_X = _df_X.filter(self.feature_importance_list)
        return _df_X

    @property
    def df_test_y(self) -> pd.DataFrame:
        """
            Target Property as pd.DataFrame
        :return:
        """
        return self.df_test.filter([self.target_variable])

    @property
    def df_val_X(self) -> pd.DataFrame:
        """
        Dataframe Property containing only the top n features from feature importance study
        """
        _df_X = self.df_val.drop(self.target_variable, axis=1, errors='ignore')
        if self.feature_importance_list:
            _df_X = _df_X.filter(self.feature_importance_list)
        return _df_X

    @property
    def df_val_y(self) -> pd.DataFrame:
        """
            Target Property as pd.DataFrame
        :return:
        """
        return self.df_val.filter([self.target_variable])

    def _process_target_col(self):
        # calculate target column, if necessary. Otherwise convert target column to the 'float' datatype
        if self.target_col_function:
            self.df_train[self.target_variable] = self.target_col_function(self.df_train)
            self.df_test[self.target_variable] = self.target_col_function(self.df_test)
            self.df_val[self.target_variable] = self.target_col_function(self.df_val)

        if self.df_train_y is not None:
            self.df[self.target_variable] = self.df_train_y.astype('float')

            self.target_stats = self.df_train_y[self.target_variable].describe()

        if self.df_test_y is not None:
            self.df[self.target_variable] = self.df_test_y.astype('float')

        if self.df_val_y is not None:
            self.df[self.target_variable] = self.df_val_y.astype('float')

    def enum_columns(self):
        """
        Enumerates any columns that contain string values.
        Accepts a dataframe and list of columns that need to be converted from string to enum.
        The return is a dataframe with the columns converted.
        """
        if self.cols_to_enum:
            for col in self.cols_to_enum:
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(self.df[[col]])
                category = self.encoders[col].classes_
                # self.encoder_mapping[col] = dict(zip(category, self.encoders[col].transform(category)))
                self.encoder_mapping[col] = dict(zip(category, list(map(int, self.encoders[col].transform(category)))))
                self.df_train[col] = self.encoders[col].transform(self.df_train[[col]])
                self.df_test[col] = self.encoders[col].transform(self.df_test[[col]])
                self.df_val[col] = self.encoders[col].transform(self.df_val[[col]])

    def decode_enum(self, df_):
        """
        Coverts data that was enumerated back to categorical data
        Accepts a dataframe and list of columns that need to be converted from numerical to str.
        The return is a dataframe with the columns converted.
        """
        if self.cols_to_enum is not None:
            for col in self.cols_to_enum:
                if col in df_.columns:  # checking if enum column is actually in the final dataframe after feature importance
                    df_[col] = pd.DataFrame(self.encoders[col].inverse_transform(df_[[col]].astype(int)), columns=[col])

        return df_

    def normalize(self):
        """
        Normalizes the df between 0 and 1.
        """
        # self.stats = self.df_train.describe()
        self.stats = self.df.describe()
        if len(self.stats.columns) != len(self.df_train.columns):
            raise ValueError("Dataframe includes string data which was not enumerated")
        for col in self.df_train.columns:
            self.column_stats[col] = {'min': self.stats[col]['min'], 'max': self.stats[col]['max']}
            self.df_train[col] = self.df_train[col].apply(
                lambda x: (x - self.stats[col]['min']) / (self.stats[col]['max'] - self.stats[col]['min']))
            self.df_test[col] = self.df_test[col].apply(
                lambda x: (x - self.stats[col]['min']) / (self.stats[col]['max'] - self.stats[col]['min']))
            self.df_val[col] = self.df_val[col].apply(
                lambda x: (x - self.stats[col]['min']) / (self.stats[col]['max'] - self.stats[col]['min']))

    # Denormalize for evaluation
    def denormalize_y(self, y):
        """
        Denormalizes the neural network's predictions for model evaluation.

        :param y: The dataset that needs to be denormalized.
        :type y: pandas Series or NumPy array
        :return: A denormalized dataset
        :rtype: pandas Series or NumPy array
        """
        return y * (self.target_stats['max'] - self.target_stats['min']) + self.target_stats['min']

    def denormalize(self, df_):
        """
        Denormalizes the input for printing predictions.
        """
        for col in df_:
            df_[col] = df_[col].apply(
                lambda x: x * (self.stats[col]['max'] - self.stats[col]['min']) + self.stats[col]['min'])

        df_ = self.decode_enum(df_)

        return df_

    def train_test_split(self, **kwargs) -> object:
        """
        Returns the Pre-Split data (training, validation and test data)
        Training and Test data are used to build model, and Validation data is reserved as unseen data to evaluate the model

        :return: a tuple of  X_train, X_test, y_train, y_test
        """

        return (self.df_train_X.values, pd.concat([self.df_val_X, self.df_test_X], axis=0, ignore_index=True).values,
                self.df_train_y.values, pd.concat([self.df_val_y, self.df_test_y], axis=0, ignore_index=True).values)

    def train_val_test_split(self, **kwargs) -> object:
        """
        Returns the Pre-Split data (training, validation and test data)
        Training and Test data are used to build model, and Validation data is reserved as unseen data to evaluate the model

        :return: a tuple of  X_train, X_val, X_test, y_train, y_val, y_test
        """

        return self.df_train_X.values, self.df_val_X.values, self.df_test_X.values, self.df_train_y.values, self.df_val_y.values, self.df_test_y.values


def read_csv(csv_file: Union[str, Path], target_variable: str,
             feature_importance: FeatureImportance = None,
             feature_list: List[str] = None,
             cols_to_enum: list[str] = None,
             # normalize_data: bool = False,
             target_col_function: Callable = None,
             train_size: float = 0.8,
             val_size: float = 0.1,
             test_size: float = 0.1,
             **kwargs) -> Data:
    """
        :param csv_file:
        :type csv_file: str
        :param target_variable:
        :type target_variable: str
        :param feature_importance: Feature Importance
        :type feature_importance: FeatureImportance = None,
        :param cols_to_enum:
        :type cols_to_enum: list[str] = None,
        :param normalize_data:
        :type normalize_data: bool = False,
        :param target_col_function:
        :type target_col_function: Callable = None,
        :param kwargs: keyword arguments passed on to pd.read_csv
        :param test_size:
        :type test_size: float = .1,
        :param val_size:
        :type val_size: float = .1,
        :param train_size:
        :type train_size: float = .8,
        :return: ampl.Data
    """
    df = pd.read_csv(csv_file, **kwargs)
    data = Data(df,
                target_variable,
                feature_importance=feature_importance,
                cols_to_enum=cols_to_enum,
                feature_list=feature_list,
                # normalize_data=normalize_data,
                target_col_function=target_col_function,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                )
    return data


def read_sql(table_name: str, db_file: str, target_variable: str,
             feature_importance: FeatureImportance = None,
             feature_list: List[str] = None,
             cols_to_enum: list[str] = None,
             # normalize_data: bool = True,
             target_col_function: Callable = None,
             train_size: float = 0.8,
             val_size: float = 0.1,
             test_size: float = 0.1,
             **kwargs) -> Data:
    """
        :param table_name:
        :type table_name: str
        :param db_file:
        :type db_file: str
        :param target_variable:
        :type target_variable: str
        :param feature_importance:
        :type feature_importance: FeatureImportance = None,
        :param cols_to_enum:
        :type cols_to_enum: list[str] = None,
        :param normalize_data:
        :type normalize_data: bool = False,
        :param target_col_function:
        :type target_col_function: Callable = None,
        :param kwargs: keyword arguments passed on to pd.read_sql_query
        :param test_size:
        :type test_size: float = .1,
        :param val_size:
        :type val_size: float = .1,
        :param train_size:
        :type train_size: float = .8,
        :return: ampl.Data
    """
    with Database.connect(db_file) as connection:
        df = pd.read_sql_query("SELECT * FROM " + table_name, connection, **kwargs)

    return Data(df,
                target_variable,
                feature_importance=feature_importance,
                cols_to_enum=cols_to_enum,
                feature_list=feature_list,
                # normalize_data=normalize_data,
                target_col_function=target_col_function,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size, )


def read_pre_split_csv(train_csv_file: str, test_csv_file: str,
                       val_csv_file: str,
                       target_variable: str,
                       feature_importance: FeatureImportance = None,
                       feature_list: List[str] = None,
                       cols_to_enum: List[str] = None,
                       target_col_function: Callable = None
                       ) -> PreSplitData:
    """

    """
    df_train = pd.read_csv(train_csv_file)
    df_test = pd.read_csv(test_csv_file)
    df_val = pd.read_csv(val_csv_file)

    data = PreSplitData(df_train, df_test, df_val, target_variable, feature_importance=feature_importance,
                        cols_to_enum=cols_to_enum, feature_list=feature_list, target_col_function=target_col_function)

    return data


def read_pre_split_sql(train_sql_table: str,
                       test_sql_table: str,
                       val_sql_table: str,
                       target_variable: str,
                       db_file: str,
                       feature_importance: FeatureImportance = None,
                       feature_list: List[str] = None,
                       cols_to_enum: List[str] = None,
                       target_col_function: Callable = None,
                       **kwargs) -> PreSplitData:
    """

    """
    with Database.connect(db_file) as connection:
        df_train = pd.read_sql_query("SELECT * FROM " + train_sql_table, connection, **kwargs)
        df_test = pd.read_sql_query("SELECT * FROM " + test_sql_table, connection, **kwargs)
        df_val = pd.read_sql_query("SELECT * FROM " + val_sql_table, connection, **kwargs)

    data = PreSplitData(df_train, df_test, df_val, target_variable, feature_importance=feature_importance,
                        cols_to_enum=cols_to_enum, feature_list=feature_list, target_col_function=target_col_function)

    return data


class SqlUtil(object):
    @staticmethod
    def load_sql_view(self):
        """
        Reads a SQL query into a DataFrame for loading a SQLite view.
        """
        with Database.connect() as connection:
            self.results_df = pd.read_sql_query("SELECT * FROM " + self.view_name, connection)

        self.results_df = self.results_df.astype('float')

    @staticmethod
    def load_sql(db: Database, table_name: str) -> pd.DataFrame:
        """
        Reads a SQL query into a Data for loading the df.
        :param db:
        :rtype: pd.DataFrame
        :param table_name:
        :type table_name: str
        """
        with db.connection() as connection:
            df = pd.read_sql_query("SELECT * FROM " + table_name, connection)

        return df

    @staticmethod
    def create_sql_view(db_file, table_name, view_name, columns):
        """
        Creates a view in SQLite that selects specific columns from a table.

        :param db_file: Path to the SQLite database file.
        :type db_file: str
        :param table_name: Name of the table to select columns from.
        :type table_name: str
        :param view_name: Name of the view to create.
        :type view_name: str
        :param columns: List of column names to select.
        :type columns: list
        """
        # create a cursor object
        with Database.connect(db_file) as connection:
            cur = connection.cursor()
            # check if the view already exists
            cur.execute("SELECT name FROM sqlite_master WHERE type='view' AND name=?",
                        (view_name,))
            view_exists = cur.fetchone() is not None
            # if the view exists, drop it
            if view_exists:
                cur.execute(f"DROP VIEW {view_name}")
            # create an SQL query to create a view of selected columns
            column_list = ', '.join(columns)
            query = f'''
            CREATE VIEW {view_name} AS
            SELECT {column_list}
            FROM {table_name}
            '''
            # execute the query to create the view
            connection.execute(query)

    @staticmethod
    def drop_first_row(db_file: str, table_name: str):
        """
        Drop the first row in a SQL table, if necessary.
        When creating a database from an excel file, sometimes the column names 
        get read into the first row of the df.
        """
        with Database.connect(db_file) as connection:
            df = pd.read_sql_query("SELECT * FROM " + table_name, connection)
            df.rename(columns=df.iloc[0], inplace=True)
            # Drop first row by selecting all rows from first row onwards
            df = df.iloc[1:, :]
            df.to_sql(table_name, connection, )

    # convert a sqlite file to csv
    @staticmethod
    def to_sql(df: pd.DataFrame, database: Database, sql_table_name: str,
               if_exists: Literal["fail", "replace", "append"] = "replace", index: bool = False) -> None:
        """
        Save a csv file to sqlite database. The user will need to provide
        the path to the csv file, the name for the sqlite database, and the Table name where
        the df will be stored.

        :param if_exists:
        :param index:
        :param database:
        :rtype: None
        :param df: Path to the CSV file
        :type df: string
        :param sql_table_name: Name for the table in the sql database
        :type sql_table_name: string
        :return: none
        """
        with database.connection() as connection:
            # Save dataframe to SQLite table
            df.to_sql(sql_table_name, connection, if_exists=if_exists, index=index)

    @staticmethod
    def convert_to_unixDatetime(sql_file_name: str, sql_table_name: str, column_name: str) -> None:
        """
        Convert a single column containing a datetime to a unix date time

        :param sql_file_name: _description_, defaults to None
        :type sql_file_name: _type_, optional
        :param sql_table_name: _description_, defaults to None
        :type sql_table_name: _type_, optional
        :param column_name: _description_, defaults to None
        :type column_name: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        if not sql_file_name or not sql_table_name or not column_name:
            print("Must have values for sql_file_name, sql_table_name, and column_name.")
            raise ValueError("Missing Variable values, Must have values for sql_file_name, sql_table_name, " +
                             "and column_name.")

        with Database.connect(sql_file_name + '.sqlite') as conn:
            # Load the df into a Pandas dataframe
            df = pd.read_sql_query('SELECT * FROM ' + sql_table_name, conn)

            # Convert datetime column to Unix timestamp
            df[column_name] = df[column_name].apply(lambda x: time.mktime(pd.to_datetime(x).timetuple()))

            # Update the SQLite table with the new df
            df.to_sql(sql_table_name, conn, if_exists='replace', index=False)

    # read sql database table into pandas dataframe and then convert the dataframe date and time column to unix datetime
    @staticmethod
    def convert_multicolumn_to_unixDatetime(sql_file_path: str, sql_table_name: str, UnixTime_column_name: str,
                                            column_name_one: str, column_name_two: str) -> None:
        """
         Convert Two columns containing a Date and Time to a Unix column by combining them.
         Store the unix datetime in a new column.
         This function could use a rewrite to update at once

        :rtype: object
        :param sql_file_path: _description_, defaults to None
        :type sql_file_path: _type_, optional
        :param sql_table_name: _description_, defaults to None
        :type sql_table_name: _type_, optional
        :param UnixTime_column_name: _description_, defaults to None
        :type UnixTime_column_name: _type_, optional
        :param column_name_one: _description_, defaults to None
        :type column_name_one: _type_, optional
        :param column_name_two: _description_, defaults to None
        :type column_name_two: _type_, optional    
        """

        with Database.connect(sql_file_path + '.sqlite') as conn:
            # Load the df into a Pandas dataframe Date, Time
            df = pd.read_sql_query('SELECT * FROM ' + sql_table_name, conn)

            # Combine Date and Time columns into a single datetime column
            df['datetime_col'] = pd.to_datetime(df[column_name_one] + ' ' + df[column_name_two])

            # Drop rows with missing or invalid dates/times
            df.dropna(subset=['datetime_col'], inplace=True)

            # Convert datetime column to Unix timestamp
            df[UnixTime_column_name] = df['datetime_col'].apply(lambda x: int(time.mktime(x.timetuple())))

            # Loop through each row in the dataframe and update the Unix timestamp column in the Air table
            for index, row in df.iterrows():
                unix_timestamp = row[UnixTime_column_name]
                date = row[column_name_one]
                t = row[column_name_two]
                conn.execute(
                    "UPDATE " + sql_table_name + " SET " + UnixTime_column_name + "=? WHERE " + column_name_one + "=? AND " + column_name_two + "=?",
                    (unix_timestamp, date, t))

    # convert a sqlite file to csv
    @staticmethod
    def drop_bad_values(sql_file_path: str, sql_table_name: str, column_name_one: str) -> None:
        """
         Convert Two columns containing a Date and Time to a Unix column by combining them.
         Store the unix datetime in a new column.
         This function could use a rewrite to update at once

        :param sql_file_path: _description_, defaults to None
        :type sql_file_path: _type_, optional
        :param sql_table_name: _description_, defaults to None
        :type sql_table_name: _type_, optional
        :param column_name_one: _description_, defaults to None
        :type column_name_one: _type_, optional
        """

        with Database.connect(sql_file_path + '.sqlite') as conn:
            # Load the df into a Pandas dataframe Date, Time
            df = pd.read_sql_query('SELECT * FROM ' + sql_table_name, conn)

            # filter the dataframe to exclude rows where column_name == value_to_drop
            df_filtered = df.drop(df[df[column_name_one] == -200].index)

            # Update the SQLite table with the new df
            df_filtered.to_sql(sql_table_name, conn, if_exists='replace', index=False)

    # convert a sqlite file to csv
    @staticmethod
    def drop_nan_values(sql_file_path: str, sql_table_name: str, column_name_one: str) -> None:
        """
         Convert Two columns containing a Date and Time to a Unix column by combining them.
         Store the unix datetime in a new column.
         This function could use a rewrite to update at once

        :rtype: None
        :param sql_file_path: _description_, defaults to None
        :type sql_file_path: _type_, optional
        :param sql_table_name: _description_, defaults to None
        :type sql_table_name: _type_, optional
        :param column_name_one: _description_, defaults to None
        :type column_name_one: _type_, optional  
        """
        with Database.connect(sql_file_path + '.sqlite') as conn:
            # Load the df into a Pandas dataframe Date, Time
            df = pd.read_sql_query('SELECT * FROM ' + sql_table_name, conn)

            # filter the dataframe to exclude rows where column_name == value_to_drop
            df.dropna(subset=[column_name_one], inplace=True)

            # Update the SQLite table with the new df
            df.to_sql(sql_table_name, conn, if_exists='replace', index=False)

    @staticmethod
    def drop_column(sql_file_path: str, sql_table_name: str, drop_column_name: str) -> None:
        """
         Convert Two columns containing a Date and Time to a Unix column by combining them.
         Store the unix datetime in a new column.
         This function could use a rewrite to update at once

        :rtype: None
        :param drop_column_name: 
        :param sql_file_path: _description_, defaults to None
        :type sql_file_path: _type_, optional
        :param sql_table_name: _description_, defaults to None
        :type sql_table_name: _type_, optional
        """
        with Database.connect(sql_file_path + '.sqlite') as conn:
            # Load the df into a Pandas dataframe Date, Time
            df = pd.read_sql_query('SELECT * FROM ' + sql_table_name, conn)

            # filter the dataframe to exclude rows where column_name == value_to_drop
            df_filtered = df.drop(drop_column_name, axis=1)

            # Update the SQLite table with the new df
            df_filtered.to_sql(sql_table_name, conn, if_exists='replace', index=False)
