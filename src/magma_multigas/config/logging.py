"""Logging configuration for magma-multigas."""

import logging
import sys
from typing import Optional

from ..core.types import LogLevel


def setup_logging(level: LogLevel = LogLevel.INFO) -> logging.Logger:
    """Setup logging with specified level.

    Args:
        level: Logging level (DEBUG, INFO, WARN, ERROR)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("magma_multigas")

    # Remove existing handlers
    logger.handlers.clear()

    # Map LogLevel to logging levels
    level_map = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.INFO: logging.INFO,
        LogLevel.WARN: logging.WARNING,
        LogLevel.ERROR: logging.ERROR,
    }

    logger.setLevel(level_map.get(level, logging.INFO))

    # Create console handler with formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level_map.get(level, logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get logger instance.

    Args:
        name: Logger name (default: magma_multigas)

    Returns:
        Logger instance
    """
    if name is None:
        name = "magma_multigas"
    return logging.getLogger(name)
