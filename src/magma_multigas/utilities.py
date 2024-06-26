import numpy as np
import pandas as pd

from typing import Dict, Tuple


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
