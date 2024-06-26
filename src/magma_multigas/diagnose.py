import pandas as pd

from .multigas_data import MultiGasData
from .query import Query, unique
from .validator import validate_mutligas_data_type, validate_column_name
from typing import Dict, List, Self


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
        return self.get()[column_name]

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
        return self.missing_values['column'].to_list()

    @property
    def availability(self) -> pd.DataFrame:
        df = self.get()
        length = len(df)
        total_empty_values = df.isna().sum().to_list()

        series: pd.Series = df.isna().any()
        series.index.name = 'column'
        series.name = 'contain_empty_data'

        new_df = pd.DataFrame(series)

        new_df['total_data_missing'] = total_empty_values
        new_df['missing_percentage'] = (new_df['total_data_missing'] / length) * 100
        new_df['total_data_available'] = length - new_df['total_data_missing']
        new_df['completeness_percentage'] = (new_df['total_data_available'] / length) * 100
        new_df = new_df.sort_values('column', ascending=True)
        new_df.reset_index(inplace=True)

        return new_df

    @property
    def completeness(self) -> pd.DataFrame:
        """Return completeness of data.

        Returns:
            pd.DataFrame: completeness of data
        """
        df = self.availability[
            ['column',
             'total_data_available',
             'completeness_percentage']
        ]

        return df

    def completeness_minimum(self, minimum: float) -> pd.DataFrame:
        """Return dataframe with minimum completeness value

        Args:
            minimum (float): minimum completeness value

        Returns:
            pd.DataFrame: dataframe with minimum completeness value
        """
        df = self.completeness
        df = (df[df['completeness_percentage'] > minimum]
              .sort_values('completeness_percentage', ascending=False))

        return df

    def completeness_maximum(self, maximum: float) -> pd.DataFrame:
        """Return dataframe with maximum completeness value

        Args:
            maximum (float): maximum completeness value

        Returns:
            pd.DataFrame: dataframe with maximum completeness value
        """
        df = self.completeness
        df = (df[df['completeness_percentage'] < maximum]
              .sort_values('completeness_percentage', ascending=False))

        return df

    @property
    def missing_values(self) -> pd.DataFrame:
        """Get missing values information

        Returns:
            pd.DataFrame: dataframe with missing values
        """
        df = self.availability[
            ['column',
             'total_data_missing',
             'missing_percentage']
        ]

        return df

    def null_percentage(self, as_dict: bool = False) -> pd.DataFrame | Dict[str, float]:
        """Get null percentage of data

        Args:
            as_dict (bool, optional): if True, return dict with null percentage

        Returns:
            pd.DataFrame: dataframe with null percentage
            Dict[str, float]: dict with null percentage
        """
        null_dict: Dict[str, float] = {}
        df = self.get()
        columns = df.columns.sort_values(ascending=True)
        for column in columns:
            null_rate = df[column].isnull().sum() / len(df) * 100
            if null_rate > 0:
                null_dict[column] = null_rate

        if as_dict is True:
            return null_dict

        return pd.DataFrame(null_dict.items(), columns=['column', 'null_percentage'])
