#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""magma-multigas v2.0 - Multi-gas volcanic monitoring data processing."""

from .core import (
    STATUSES,
    DatasetError,
    DatasetType,
    FileFormat,
    LogLevel,
    MagmaMultigasError,
    ValidationError,
)
from .data import Dataset, DatasetCollection, DatasetMetadata

# v2.0 API (preferred)
from .analysis import DataDiagnostics
from .multigas import MultiGas
from .plotting import AvailabilityPlotter, PlotConfig, TimeSeriesPlotter

# Backward compatibility with v1.x (imports from old module)
try:
    from magma_multigas_old.diagnose import Diagnose, Query
    from magma_multigas_old.multigas_data import MultiGasData
    from magma_multigas_old.plot_availability import PlotAvailability
    from magma_multigas_old.plot_var import PlotWithMagma
    from magma_multigas_old.plot_wind_direction import PlotWindDirection

    _V1_AVAILABLE = True
except ImportError:
    _V1_AVAILABLE = False
    MultiGasData = None
    Diagnose = None
    Query = None
    PlotAvailability = None
    PlotWindDirection = None
    PlotWithMagma = None

try:
    from importlib.metadata import version

    __version__ = version("magma-multigas")
except Exception:
    __version__ = "2.0.0"

__author__ = "Martanto"
__author_email__ = "martanto@LIVE.COM"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024-2026, Martanto"
__url__ = "https://github.com/martanto/magma-multigas"

# v2.0 exports (primary API)
__all__ = [
    # Core classes
    "MultiGas",
    "Dataset",
    "DatasetCollection",
    "DatasetMetadata",
    # Analysis
    "DataDiagnostics",
    # Plotting
    "PlotConfig",
    "TimeSeriesPlotter",
    "AvailabilityPlotter",
    # Types and enums
    "DatasetType",
    "LogLevel",
    "FileFormat",
    # Exceptions
    "MagmaMultigasError",
    "DatasetError",
    "ValidationError",
    # Constants
    "STATUSES",
]

# v1.x backward compatibility (conditionally exported)
if _V1_AVAILABLE:
    __all__.extend(
        [
            "MultiGasData",
            "Diagnose",
            "Query",
            "PlotAvailability",
            "PlotWindDirection",
            "PlotWithMagma",
        ]
    )
