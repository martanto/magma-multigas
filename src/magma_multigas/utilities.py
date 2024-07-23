import numpy as np
import pandas as pd

from typing import Dict, Tuple
from .variables import wind_direction_8, wind_direction_16


def regression_function(df: pd.DataFrame, column: str) -> str:
    _, slope, intercept = get_slope_and_intercept(df, column)
    return f'y = {slope:.2f}x + {intercept:.2f}'


def get_slope_and_intercept(df: pd.DataFrame, column: str) -> Tuple[np.ndarray, float, float]:
    """Calculate the slope and intercept of the linear regression.

    Args:
        df (pd.DataFrame): Dataframe index act as 'x'.
        column (str): The column name. Act as 'y'

    Returns:
        Tuple[float, float]: slope and intercept.
    """
    x = df.index
    if isinstance(x, pd.DatetimeIndex):
        x = np.arange(len(df.index))

    y = df[column]

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean
    return x, slope, intercept


def y_prediction(df: pd.DataFrame, column: str) -> np.ndarray:
    x, slope, intercept = get_slope_and_intercept(df, column)
    return slope * x + intercept


def mean_squared_error(y_true, y_pred) -> np.float64:
    return np.sum(np.square(y_pred - np.mean(y_true))) / len(y_true)


def root_mean_squared_error(y_true, y_pred) -> np.float64:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r_squared(y_true, y_pred) -> np.float64:
    return np.sum(np.square(y_pred - np.mean(y_true))) / np.sum(np.square(y_true - np.mean(y_true)))


def all_evaluations(df: pd.DataFrame, column: str) -> Dict[str, np.float64]:
    y_true = df[column]
    y_pred = y_prediction(df, column)

    return {
        'mean_squared_error': mean_squared_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'r2': r_squared(y_true, y_pred)
    }


def convert_to_direction(direction_degree: float, return_as_code: bool = False, direction_to_use: int = 16) -> str:
    """Convert degree to wind direction.

    Args:
        direction_degree (float): The degree to convert.
        return_as_code (bool, optional): Whether to return code or not. Defaults to False.
        direction_to_use (int, optional): Which direction to convert. Defaults to 16.

    Returns:
        str: The converted direction.
    """
    assert (direction_to_use == 16) or (direction_to_use == 8), ValueError("direction_to_use must be either 16 or 8")

    wind_directions = wind_direction_16 if (direction_to_use == 16) else wind_direction_8

    for directions in wind_directions:
        if directions['min_degree'] <= direction_degree < directions['max_degree']:
            if return_as_code is True:
                return directions['code']
            return directions['direction']
