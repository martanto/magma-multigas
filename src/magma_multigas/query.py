from .validator import validate_status, validate_column_name, validate_comparator
from typing import List, Any, Self

import pandas as pd
from datetime import timedelta


class Query:
    def __init__(self, df: pd.DataFrame):
        """Query data

        Args:
            df (pd.DataFrame): data frame

        Attributes:
            df (pd.DataFrame): data frame
        """
        self.df: pd.DataFrame = df
        self.columns: List[str] = df.columns.tolist()
        self.columns_selected: List[str] = []
        self.start_date: str = df.index[0].strftime('%Y-%m-%d')
        self.end_date: str = df.index[-1].strftime('%Y-%m-%d')

    def translate_comparator(self, column_name: str, comparator: str, value: Any) -> pd.DataFrame:
        """Translate comparator

        Args:
            column_name (str): column name
            comparator (str): comparator
            value (Any): value

        Returns:
            pd.DataFrame: data frame
        """
        validate_comparator(comparator)

        df = self.df.copy()

        column = df[column_name]
        if column_name == df.index.name:
            column = df.index

        if comparator in ['==', 'like', 'equal', 'eq', 'sama dengan']:
            return df[column == value]
        if comparator in ['!=', 'ne', 'not equal', 'tidak sama dengan']:
            return df[column != value]
        if comparator in ['>', 'gt', 'greater than', 'lebih besar', 'lebih besar dari']:
            return df[column > value]
        if comparator in ['<', 'lt', 'less than', 'kurang', 'kurang dari']:
            return df[column < value]
        if comparator in ['>=', 'gte', 'greater than equal', 'lebih besar sama dengan']:
            return df[column >= value]
        if comparator in ['<=', 'lte', 'less than equal', 'kurang dari sama dengan']:
            return df[column <= value]
        self.df = df
        return self.df

    def count(self) -> int:
        """Count number of data

        Returns:
            int: number of data
        """
        return len(self.get())

    def select_columns(self, column_names: str | List[str]) -> Self:
        """Select columns

        Args:
            column_names (str | List(str)): column names

        Returns:
            self (Self)
        """
        if isinstance(column_names, str):
            column_names = [column_names]

        for column_name in column_names:
            validate_column_name(column_name, self.columns)

        self.columns_selected = column_names
        return self

    def where(self, column_name: str, comparator: str, value: Any) -> Self:
        """Filter data based on column value

        Args:
            column_name (str): column name
            comparator (str): comparator
            value (Any): value

        Returns:
            self (Self)
        """
        if column_name == 'Status_Flag':
            self.where_status(value)

        column_list: List[str] = self.columns
        validate_column_name(column_name, column_list)

        self.df = self.translate_comparator(column_name, comparator, value)
        return self

    def where_status(self, value: Any) -> Self:
        """Filter status Flag

        Args:
            value (Any): Status value

        Returns:
            self (Self)
        """
        validate_status(int(value))
        self.df = self.translate_comparator('Status_Flag', '==', value)
        return self

    def where_date(self, date_str: str) -> Self:
        """Filter data based on date string

        Args:
            date_str (str): date string with format YYYY-MM-DD
        """
        self.df = self.df.loc[date_str]
        return self

    def where_values_between(self, column_name: str, start_value: int | float, end_value: int | float) -> Self:
        """Filter data based on two values in specified column

        Args:
            column_name (str): column name
            start_value (int | float): start value
            end_value (int | float): end value

        Returns:
            self (Self)
        """
        df = self.df
        self.df = df[df[column_name].between(start_value, end_value)]
        return self

    def where_date_between(self, start_date: str, end_date: str) -> Self:
        """Filter data based on start and end date

        Args:
            start_date (str): start date. Date format YYYY-MM-DD
            end_date (str): end date. Date format YYYY-MM-DD

        Returns:
            self (Self)
        """
        self.start_date = start_date
        self.end_date = end_date

        start_date = pd.to_datetime(start_date, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date, format='%Y-%m-%d') + timedelta(days=1)

        df = self.df
        self.df: pd.DataFrame = df[(df.index >= start_date) & (df.index <= end_date)]
        return self

    def get(self) -> pd.DataFrame:
        """Get filtered data

        Returns:
            pd.DataFrame: filtered data
        """
        if len(self.columns_selected) == 0:
            return self.df

        return self.df[self.columns_selected]
