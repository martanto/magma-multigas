"""Collection of multiple datasets."""

from typing import Dict, Iterator, Optional

import pandas as pd

from ..core.exceptions import DatasetError
from ..core.types import DatasetType, DateLike
from .dataset import Dataset


class DatasetCollection:
    """Manages multiple datasets (two_seconds, six_hours, etc.).

    Provides dictionary-like access to datasets with convenient
    methods for filtering all datasets at once.

    Attributes:
        datasets: Dictionary mapping dataset type to Dataset instance
    """

    def __init__(self, datasets: Dict[str, Dataset]):
        """Initialize collection with datasets.

        Args:
            datasets: Dictionary of dataset_type -> Dataset
        """
        self._datasets = datasets

    def __getitem__(self, key: str) -> Dataset:
        """Get dataset by type name.

        Args:
            key: Dataset type name (e.g., 'six_hours')

        Returns:
            Dataset instance

        Raises:
            KeyError: If dataset type doesn't exist
        """
        if key not in self._datasets:
            available = list(self._datasets.keys())
            raise KeyError(
                f"Dataset type '{key}' not found. Available types: {available}"
            )
        return self._datasets[key]

    def __contains__(self, key: str) -> bool:
        """Check if dataset type exists.

        Args:
            key: Dataset type name

        Returns:
            True if dataset exists
        """
        return key in self._datasets

    def __iter__(self) -> Iterator[str]:
        """Iterate over dataset type names."""
        return iter(self._datasets)

    def __len__(self) -> int:
        """Get number of datasets in collection."""
        return len(self._datasets)

    def __repr__(self) -> str:
        """String representation of collection."""
        types = list(self._datasets.keys())
        return f"DatasetCollection(types={types})"

    def keys(self):
        """Get dataset type names."""
        return self._datasets.keys()

    def values(self):
        """Get Dataset instances."""
        return self._datasets.values()

    def items(self):
        """Get (type, Dataset) pairs."""
        return self._datasets.items()

    def get(self, key: str, default: Optional[Dataset] = None) -> Optional[Dataset]:
        """Get dataset by type with default.

        Args:
            key: Dataset type name
            default: Default value if key doesn't exist

        Returns:
            Dataset instance or default
        """
        return self._datasets.get(key, default)

    def filter_all(
        self,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        inclusive: str = "both",
    ) -> "DatasetCollection":
        """Filter all datasets by date range.

        Convenience method to apply same date filter to all datasets.

        Args:
            start: Start date
            end: End date
            inclusive: Include boundaries ('both', 'left', 'right', 'neither')

        Returns:
            New DatasetCollection with filtered datasets
        """
        filtered_datasets = {}

        for dataset_type, dataset in self._datasets.items():
            try:
                filtered = dataset.filter_date_range(start, end, inclusive)
                filtered_datasets[dataset_type] = filtered
            except Exception as e:
                # Log warning but continue with other datasets
                import logging

                logger = logging.getLogger("magma_multigas")
                logger.warning(
                    f"Failed to filter {dataset_type}: {e}. Skipping this dataset."
                )

        return DatasetCollection(filtered_datasets)

    def to_dict(self) -> Dict[str, pd.DataFrame]:
        """Convert collection to dictionary of DataFrames.

        Returns:
            Dictionary mapping dataset type to DataFrame
        """
        return {
            dataset_type: dataset.df for dataset_type, dataset in self._datasets.items()
        }

    def summary(self) -> pd.DataFrame:
        """Get summary statistics for all datasets.

        Returns:
            DataFrame with summary info for each dataset
        """
        summary_data = []

        for dataset_type, dataset in self._datasets.items():
            start, end = dataset.date_range
            summary_data.append(
                {
                    "dataset_type": dataset_type,
                    "rows": len(dataset),
                    "columns": len(dataset.columns),
                    "start_date": start,
                    "end_date": end,
                    "station": dataset.metadata.station,
                }
            )

        return pd.DataFrame(summary_data)
