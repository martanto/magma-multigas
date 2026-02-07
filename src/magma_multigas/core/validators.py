"""Validation utilities for magma-multigas v2.0.

Migrated from v1.x with fixes:
- Removed print statement on line 127
- Removed emoji from error messages
- Added type hints improvements
"""

from datetime import date, datetime
from typing import Any

import pandas as pd

STATUS_NO: list[int] = [-2, -1, 0, 1, 4, 6, 10, 11, 14, 16]
STATUS_DESCRIPTION: list[str] = [
    "Chemical Sensor Off",
    "Warming Up",
    "Zero",
    "Sample Acquisition",
    "Standart Gas Measurement for CO2 and SO2",
    "Standart Gas Measurement for H2S",
    "Manual Zero Measurement",
    "Manual Sample Measurement",
    "Manual Standart Gas Measurement for CO2 and SO2",
    "Manual Standart Gas Measurement for H2S",
]

COMPARATORS: tuple = (
    "==",
    "like",
    "equal",
    "eq",
    "sama dengan",
    "!=",
    "ne",
    "not equal",
    "tidak sama dengan",
    ">",
    "gt",
    "greater than",
    "lebih besar",
    "lebih besar dari",
    "<",
    "lt",
    "less than",
    "kurang",
    "kurang dari",
    ">=",
    "gte",
    "greater than equal",
    "lebih besar sama dengan",
    "<=",
    "lte",
    "less than equal",
    "kurang dari sama dengan",
)

TYPE_OF_DATA: tuple = ("two_seconds", "six_hours", "one_minute", "zero", "span")

STATUSES: list[tuple[int, str]] = list(zip(STATUS_NO, STATUS_DESCRIPTION))


def validate_file_type(file_type: str) -> bool:
    """Validate file type for saving.

    Args:
        file_type: File type string (csv, excel, xlsx, xls, parquet, json)

    Returns:
        True if valid

    Raises:
        ValueError: If file type is not supported
    """
    if file_type.lower() not in ["csv", "excel", "xlsx", "xls", "parquet", "json"]:
        raise ValueError(
            f"Unsupported file type: {file_type}. "
            f'Please choose from "csv", "excel", "xlsx", "xls", "parquet", "json"'
        )
    return True


def validate_selected_data(select_data: str) -> bool:
    """Validate selected data type.

    Args:
        select_data: Data type string

    Returns:
        True if valid

    Raises:
        ValueError: If data type is invalid
    """
    if select_data.lower() not in TYPE_OF_DATA:
        raise ValueError(f"Data selected must be one of {TYPE_OF_DATA}")
    return True


def validate_comparator(comparator: str) -> bool:
    """Validate comparator operator.

    Args:
        comparator: Comparator string

    Returns:
        True if valid

    Raises:
        ValueError: If comparator is invalid
    """
    if comparator not in COMPARATORS:
        raise ValueError(
            f"Invalid comparator: {comparator}. Valid comparators are {COMPARATORS}"
        )
    return True


def in_values(column_name: str, value: Any, list_value: Any) -> bool:
    """Validate that column value exists in list of values.

    Args:
        column_name: Name of the column
        value: Value to check
        list_value: List of valid values

    Returns:
        True if valid

    Raises:
        ValueError: If value is not in list
    """
    if value not in list_value:
        raise ValueError(
            f"Value: {value} of column {column_name} must be in {list_value}"
        )
    return True


def validate_column_name(column_name: str, column_list: list[str]) -> bool:
    """Validate that column name exists in column list.

    Args:
        column_name: Column name to check
        column_list: List of valid column names

    Returns:
        True if valid

    Raises:
        ValueError: If column is not found
    """
    if column_name not in column_list:
        raise ValueError(f"Column {column_name} is not found in {column_list}")
    return True


def validate_status(status_value: int, status_column: str | None = None) -> bool:
    """Validate status value.

    Args:
        status_value: Status value to check
        status_column: Name of the status column (default: 'Status_Flag')

    Returns:
        True if valid

    Raises:
        ValueError: If status value is invalid
    """
    if status_column is None:
        status_column = "Status_Flag"

    if in_values(status_column, status_value, STATUS_NO) is True:
        return True

    # Fixed: Removed print(stat) statement
    raise ValueError(f"Status value must be in {STATUS_NO}")


def validate_date(date_str: str) -> bool:
    """Validate date format (yyyy-mm-dd).

    Args:
        date_str: Date string to validate

    Returns:
        True if valid

    Raises:
        ValueError: If date format is invalid
    """
    try:
        date.fromisoformat(date_str)
        return True
    except ValueError:
        raise ValueError("Incorrect date format, should be yyyy-mm-dd")


def validate_datetime(datetime_str: str) -> bool:
    """Validate datetime format (yyyy-mm-dd or yyyy-mm-dd HH:MM:SS).

    Args:
        datetime_str: Datetime string to validate

    Returns:
        True if valid

    Raises:
        ValueError: If datetime format is invalid
    """
    try:
        datetime.fromisoformat(datetime_str)
        return True
    except ValueError:
        raise ValueError(
            "Incorrect date time format, should be yyyy-mm-dd or yyyy-mm-dd HH:MM:SS"
        )


def validate_index_as_datetime(df: pd.DataFrame) -> bool:
    """Validate that DataFrame index is DatetimeIndex.

    Args:
        df: DataFrame to check

    Returns:
        True if valid

    Raises:
        ValueError: If index is not DatetimeIndex
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return True
    raise ValueError("Index is not valid pd.DatetimeIndex")


def validate_multigas_data_type(multigas_data_type: str) -> bool:
    """Validate multigas data type.

    Args:
        multigas_data_type: Data type to validate

    Returns:
        True if valid

    Raises:
        ValueError: If data type is invalid
    """
    return validate_selected_data(multigas_data_type)
