"""Type definitions and enums for magma-multigas v2.0."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TypedDict, Union

import pandas as pd


class DatasetType(str, Enum):
    """Types of datasets supported by magma-multigas."""

    TWO_SECONDS = "two_seconds"
    SIX_HOURS = "six_hours"
    ONE_MINUTE = "one_minute"
    ZERO = "zero"
    SPAN = "span"


class LogLevel(str, Enum):
    """Logging verbosity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class FileFormat(str, Enum):
    """Supported output file formats."""

    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"
    JSON = "json"


# Type aliases for better readability
DateLike = Union[str, datetime, pd.Timestamp]
ColumnName = str
Comparator = str  # e.g., "==", "!=", ">", "<", ">=", "<="
PathLike = Union[str, Path]


class DatasetMetadataDict(TypedDict, total=False):
    """Metadata extracted from dataset files."""

    station: str
    logger_type: str
    firmware: str
    program_name: str
    file_sampling: str
    serial_number: str
    os_version: str


class CacheMetadataDict(TypedDict):
    """Metadata for cache validation."""

    file_path: str
    mtime: float
    normalize: bool
    checksum: str


class PlotConfigDict(TypedDict, total=False):
    """Configuration for plotting."""

    width: int
    height: int
    dpi: int
    style: str
    color_scheme: str
    title: str
    xlabel: str
    ylabel: str
    legend: bool
    grid: bool
