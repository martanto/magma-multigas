from typing import List, Self

import pandas as pd

from .multigas_data import MultiGasData
from .query import Query, unique
from .validator import validate_mutligas_data_type, validate_column_name

COLUMNS: List[str] = [
    'Licor_volts',
    'Licor_bench_temp',
    'Licor_pressure',
]


class Diagnose(Query):
    def __init__(self, data: MultiGasData, columns: List[str] = None):
        type_of_data = data.type_of_data
        validate_mutligas_data_type(type_of_data)

        super().__init__(data.df)

        if columns is None:
            columns = data.columns

        if (columns is None) and (type_of_data == 'two_seconds'):
            columns = COLUMNS

        if columns is not None:
            for column in columns:
                validate_column_name(column, data.columns)
            if type_of_data == 'two_seconds':
                columns = unique(columns, COLUMNS)
            self.select_columns(columns)

        self.default_columns = columns
        self.type_of_data = type_of_data

    def __str__(self) -> str:
        """Return type of diagnosis data"""
        return self.describe()

    def __getattr__(self, column_name: str) -> pd.DataFrame:
        """Get dataframe, when columns is not available"""
        return self.df_original[column_name]

    def describe(self) -> str:
        """Describe class"""
        return (f"{type(self).__name__}(type_of_data={self.type_of_data}, "
                f"length={self.count()}, start_date={self.start_datetime}, "
                f"end_date={self.end_datetime}, columns_selected={self.columns_selected})")

    def add_columns(self, columns: str | List[str]) -> Self:
        """Add another column(s) to check

        Args:
            columns (str | List[str]): column(s) to add

        Returns:
            Self: self
        """
        if isinstance(columns, str):
            columns = [columns]

        self.default_columns = unique(columns, self.default_columns)
        self.select_columns(self.default_columns)
        return self

    @property
    def columns_with_missing_values(self) -> List[str]:
        """Get columns with missing values

        Returns:
            List[str]: columns with missing values
        """
        return self.missing_values.index.to_list()

    @property
    def missing_values(self) -> pd.DataFrame:
        """Get missing values information

        Returns:
            pd.DataFrame: dataframe with missing values
        """
        df = self.get()
        length = len(df)

        series: pd.Series = df.isna().any()
        series.index.name = 'columns'
        series.name = 'contain_empty_data'

        new_df = pd.DataFrame(series)
        new_df['total_missing_values'] = df.isna().sum().to_list()
        new_df['completeness'] = (length-new_df['total_missing_values'])/length*100
        new_df = new_df[new_df['contain_empty_data'] == True].sort_values('completeness', ascending=False)

        return new_df

    def minimum_completeness(self, completeness: float) -> pd.DataFrame:
        """Return dataframe with minimum completeness value

        Args:
            completeness (float): minimum completeness value

        Returns:
            pd.DataFrame: dataframe with minimum completeness value
        """
        df = self.missing_values
        df = df[df['completeness'] > completeness].sort_values('completeness', ascending=False)

        return df

    def maximum_completeness(self, completeness: float) -> pd.DataFrame:
        """Return dataframe with maximum completeness value

        Args:
            completeness (float): maximum completeness value

        Returns:
            pd.DataFrame: dataframe with maximum completeness value
        """
        df = self.missing_values
        df = df[df['completeness'] < completeness].sort_values('completeness', ascending=False)

        return df
