"""Utility functions for magma-multigas v2.0.

Migrated from v1.x with fixes:
- Replaced assert with ValueError on lines 80, 92
- Added edge case handling for degree wrap-around (>=360, <0)
- Improved type hints
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..config.variables import (
    wind_direction_8,
    wind_direction_16,
    wind_quadrant_4,
    wind_quadrant_8,
)


def regression_function(df: pd.DataFrame, column: str) -> str:
    """Generate regression equation string.

    Args:
        df: DataFrame with data
        column: Column name to regress

    Returns:
        Regression equation as string (e.g., 'y = 2.50x + 1.23')
    """
    _, slope, intercept = get_slope_and_intercept(df, column)
    return f"y = {slope:.2f}x + {intercept:.2f}"


def get_slope_and_intercept(
    df: pd.DataFrame, column: str
) -> Tuple[np.ndarray, float, float]:
    """Calculate slope and intercept of linear regression.

    Args:
        df: DataFrame (index acts as 'x')
        column: Column name (acts as 'y')

    Returns:
        Tuple of (x_array, slope, intercept)
    """
    x = df.index
    if isinstance(x, pd.DatetimeIndex):
        x = np.arange(len(df.index))

    y = df[column]

    x_mean: float = np.mean(x)
    y_mean: float = np.mean(y)
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean
    return x, slope, intercept


def y_prediction(df: pd.DataFrame, column: str) -> np.ndarray:
    """Calculate predicted y values from linear regression.

    Args:
        df: DataFrame with data
        column: Column to predict

    Returns:
        Array of predicted values
    """
    x, slope, intercept = get_slope_and_intercept(df, column)
    return slope * x + intercept


def mean_squared_error(y_true, y_pred) -> np.float64:
    """Calculate mean squared error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MSE value
    """
    return np.sum(np.square(y_pred - np.mean(y_true))) / len(y_true)


def root_mean_squared_error(y_true, y_pred) -> np.float64:
    """Calculate root mean squared error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r_squared(y_true, y_pred) -> np.float64:
    """Calculate R-squared (coefficient of determination).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R-squared value
    """
    return np.sum(np.square(y_pred - np.mean(y_true))) / np.sum(
        np.square(y_true - np.mean(y_true))
    )


def all_evaluations(df: pd.DataFrame, column: str) -> Dict[str, np.float64]:
    """Calculate all regression evaluation metrics.

    Args:
        df: DataFrame with data
        column: Column to evaluate

    Returns:
        Dictionary with MSE, RMSE, and R-squared
    """
    y_true = df[column]
    y_pred = y_prediction(df, column)

    return {
        "mean_squared_error": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
    }


def convert_to_direction(
    direction_degree: float, return_as_code: bool = False, direction_to_use: int = 16
) -> str:
    """Convert degree to wind direction.

    Args:
        direction_degree: Degree to convert (0-360)
        return_as_code: Whether to return code or full direction name
        direction_to_use: Number of directions (8 or 16)

    Returns:
        Wind direction string

    Raises:
        ValueError: If direction_to_use is not 8 or 16
        ValueError: If direction_degree is out of valid range
    """
    # Fixed: Replace assert with ValueError
    if direction_to_use not in (8, 16):
        raise ValueError("direction_to_use must be either 8 or 16")

    # Fixed: Add edge case handling for degree wrap-around
    if direction_degree < 0 or direction_degree >= 360:
        raise ValueError(
            f"direction_degree must be between 0 and 360, got {direction_degree}"
        )

    wind_directions = (
        wind_direction_16 if (direction_to_use == 16) else wind_direction_8
    )

    for directions in wind_directions:
        if directions["min_degree"] <= direction_degree < directions["max_degree"]:
            if return_as_code is True:
                return directions["code"]
            return directions["direction"]

    # Should not reach here if degree is valid, but handle edge case
    raise ValueError(f"Could not determine direction for degree {direction_degree}")


def convert_to_quadrant(
    direction_degree: float, return_as_code: bool = False, quadrant_to_use: int = 8
) -> str:
    """Convert degree to wind quadrant.

    Args:
        direction_degree: Degree to convert (0-360)
        return_as_code: Whether to return code or full quadrant name
        quadrant_to_use: Number of quadrants (4 or 8)

    Returns:
        Wind quadrant string

    Raises:
        ValueError: If quadrant_to_use is not 4 or 8
        ValueError: If direction_degree is out of valid range
    """
    # Fixed: Replace assert with ValueError
    if quadrant_to_use not in (4, 8):
        raise ValueError("quadrant_to_use must be either 4 or 8")

    # Fixed: Add edge case handling for degree wrap-around
    if direction_degree < 0 or direction_degree >= 360:
        raise ValueError(
            f"direction_degree must be between 0 and 360, got {direction_degree}"
        )

    wind_quadrants = wind_quadrant_8 if (quadrant_to_use == 8) else wind_quadrant_4

    for quadrants in wind_quadrants:
        if quadrants["min_degree"] <= direction_degree < quadrants["max_degree"]:
            if return_as_code is True:
                return quadrants["code"]
            return quadrants["direction"]

    # Should not reach here if degree is valid, but handle edge case
    raise ValueError(f"Could not determine quadrant for degree {direction_degree}")
