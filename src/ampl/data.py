# df.py
from dataclasses import dataclass, field
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Union, Optional, Literal, List, Any
from sklearn.preprocessing import OrdinalEncoder

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


@dataclass
class Data:
    """
    Data class encapsulates all the input data related functionalities, functions such as IO operations,
    normalization, standardization, and feature importance study
    """
    df: pd.DataFrame = field(compare=False, repr=False)
    """ Main dataframe with the features (X) and target variable (y) in Pandas dataframe format """

    target_variable: str
    """ target variable (y) column name"""

    feature_importance: FeatureImportance = field(default_factory=lambda: FeatureImportance(), repr=False)
    """ Feature Importance to run feature importance study using AutoML(XGBoost)  """

    cols_to_enum: List[str] = field(default_factory=list)
    """ Columns to enumerate, takes the identified categorical columns and converts it to ordinal(numerical)"""
    #
    # normalize_data: bool = True
    # """ Is the data normalized, if True(default) then normalization won't be applied, otherwise it will be applied """

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


    def __post_init__(self):
        if not isinstance(self.df, pd.DataFrame):
            raise ValueError('Field "df" must be of Pandas Dataframe type and cannot be None')
        self._verify_target_variable()
        self._transform_data()

        assert 1.0 == (self.train_size + self.val_size + self.test_size), \
            'Training, test, and validation size (percentages) does not add up to 1 (100 %)'

    @property
    def df_X(self) -> pd.DataFrame:
        _df_X = self.df.drop(self.target_variable, axis=1, errors='ignore')
        if self.feature_importance_list:
            _df_X = _df_X.filter(self.feature_importance_list)
        return _df_X

    @property
    def df_y(self) -> pd.DataFrame:
        """

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
        if self.cols_to_enum:
            self.enum_columns()

        # calculate target column, if necessary. Otherwise convert target column to the 'float' datatype
        if self.target_col_function:
            self.df[self.target_variable] = self.target_col_function(self.df_X)
            
        if self.df_y is not None:
            self.df[self.target_variable] = self.df_y.astype('float')

            self.target_stats = self.df_y[self.target_variable].describe()

        # Normalize df between 0 and 1
        self.normalize()

        if self.feature_importance:
            self.feature_importance_list = self.feature_importance.run(self.df_X, self.df_y.squeeze())

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
        self.encoders = {}
        for col in self.cols_to_enum:
            self.encoders[col] = OrdinalEncoder().set_output(transform="pandas")
            self.df[col] = self.encoders[col].fit_transform(self.df[[col]])
        
    def decode_enum(self, df_):
        """
        Coverts data that was enumerated back to categorical data
        Accepts a dataframe and list of columns that need to be converted from numerical to str.
        The return is a dataframe with the columns converted.
        """
        for col in self.cols_to_enum:
            if col in df_.columns: # checking if enum column is actually in the final dataframe after feature importance
                df_[col] = pd.DataFrame(self.encoders[col].inverse_transform(df_[[col]]), columns=[col])
            
        return df_

    def normalize(self):
        """
        Normalizes the df between 0 and 1.
        """
        self.stats = self.df.describe()
        if len(self.stats.columns) != len(self.df.columns):
            raise ValueError("Dataframe includes string data which was not enumerated")
        for col in self.df:
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

def read_csv(csv_file: Union[str, Path], target_variable: str,
             feature_importance: FeatureImportance = None,
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
                # normalize_data=normalize_data,
                target_col_function=target_col_function,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                )
    return data


def read_sql(table_name: str, db_file: str, target_variable: str,
             feature_importance: FeatureImportance = None,
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
                # normalize_data=normalize_data,
                target_col_function=target_col_function,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size, )


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
