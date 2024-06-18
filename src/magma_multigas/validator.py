from typing import Tuple, List, Any
from datetime import date


STATUS_NO: list[int] = [0, 1, 4, 6, 10, 11, 14, 16]
STATUS_DESCRIPTION: list[str] = [
    'Zero',
    'Sample Acquisition',
    'Standart Gas Measurement for CO2 and SO2',
    'Standart Gas Measurement for H2S',
    'Manual Zero Measurement',
    'Manual Sample Measurement',
    'Manual Standart Gas Measurement for CO2 and SO2',
    'Manual Standart Gas Measurement for H2S'
]

COMPARATORS: tuple = (
    '==', 'like', 'equal', 'eq', 'sama dengan',
    '!=', 'ne', 'not equal', 'tidak sama dengan',
    '>', 'gt', 'greater than', 'lebih besar', 'lebih besar dari',
    '<', 'lt', 'less than', 'kurang', 'kurang dari',
    '>=', 'gte', 'greater than equal', 'lebih besar sama dengan',
    '<=', 'lte', 'less than equal', 'kurang dari sama dengan'
)

TYPE_OF_DATA: tuple = (
    'two_seconds', 'six_hours', 'one_minute', 'zero', 'span'
)

STATUSES: List[Tuple[int, str]] = list(zip(STATUS_NO, STATUS_DESCRIPTION))


def validate_file_type(file_type: str) -> bool | Exception:
    """Validating file type to save

    Args:
        file_type (str): Chose between 'csv', 'excel', 'xlsx', 'xls'
    """
    if file_type.lower() not in ['csv', 'excel', 'xlsx', 'xls']:
        raise ValueError(f'Unsupported file type: {file_type}. '
                         f'Please choose from "csv", "excel", "xlsx", "xls"')
    return True


def validate_select_data(select_data: str) -> bool | Exception:
    """Validating selected data.

    Args:
        select_data (str): Selected data.

    Returns:
        True or raise an Exception.
    """
    if select_data.lower() not in TYPE_OF_DATA:
        raise ValueError('Data selected must be one of {}'.format(TYPE_OF_DATA))
    return True


def validate_comparator(comparator: str) -> bool | Exception:
    """Validating comparator.

    Args:
        comparator (str): Comparator.

    Returns:
        True or raise an Exception.
    """
    if comparator not in COMPARATORS:
        raise ValueError('⛔ Invalid comparator: {}. Valid comparators are {}'.format(comparator, COMPARATORS))
    return True


def in_values(column_name: str, value: Any, list_value: Any) -> bool | Exception:
    """Validating column value exists in another values.

    Args:
        column_name (str): Column name.
        value (Any): Value of column.
        list_value (Any): Value of list value.

    Returns:
        True or raise an Exception.
    """
    if value not in list_value:
        raise ValueError("⛔ Value: {} of column {} must be in {}".format(value, column_name, list_value))
    return True


def validate_column_name(column_name: str, column_list: list[str]) -> bool | Exception:
    """Validating column name.

    Args:
        column_name (str): Column name.
        column_list (list): List of column names.

    Returns:
        True or raise an Exception.
    """
    if column_name not in column_list:
        raise ValueError("⛔ Column {} is not found in {}".format(column_name, column_list))
    return True


def validate_status(status_value: int) -> bool | Exception:
    """Validating status.

    Args:
        status_value (int): Status value.

    Returns:
        True or raise an Exception.
    """
    if in_values('Status_Flag', status_value, STATUS_NO) is True:
        return True

    for stat in STATUSES:
        print(stat)
    raise ValueError("⛔ Status value must be in {}".format(STATUS_NO))


def validate_date(date_str: str) -> bool | Exception:
    """Validating date format.

    Args:
        date_str (str): Date format.

    Returns:
        True or raise an Exception.
    """
    try:
        date.fromisoformat(date_str)
        return True
    except ValueError:
        raise ValueError("⛔ Incorrect data format, should be YYYY-MM-DD")
