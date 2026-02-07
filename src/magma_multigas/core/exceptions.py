"""Custom exceptions for magma-multigas v2.0."""


class MagmaMultigasError(Exception):
    """Base exception for all magma-multigas errors."""

    pass


class DatasetError(MagmaMultigasError):
    """Raised when there are issues with dataset operations."""

    pass


class ValidationError(MagmaMultigasError):
    """Raised when data validation fails."""

    pass


class CacheError(MagmaMultigasError):
    """Raised when cache operations fail."""

    pass


class LoaderError(MagmaMultigasError):
    """Raised when file loading fails."""

    pass


class MetadataError(MagmaMultigasError):
    """Raised when metadata extraction or validation fails."""

    pass


class FilterError(DatasetError):
    """Raised when filtering operations fail."""

    pass


class ColumnError(DatasetError):
    """Raised when column operations fail (missing, invalid, etc.)."""

    pass


class DateRangeError(DatasetError):
    """Raised when date range operations fail."""

    pass


class PlotError(MagmaMultigasError):
    """Raised when plotting operations fail."""

    pass


class ConfigError(MagmaMultigasError):
    """Raised when configuration is invalid."""

    pass
