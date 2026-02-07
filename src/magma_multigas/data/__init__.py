"""Data loading, metadata extraction, and dataset management."""

from .collection import DatasetCollection
from .dataset import Dataset
from .loader import DataLoader
from .metadata import DatasetMetadata, MetadataExtractor

__all__ = [
    "DataLoader",
    "DatasetMetadata",
    "MetadataExtractor",
    "Dataset",
    "DatasetCollection",
]
