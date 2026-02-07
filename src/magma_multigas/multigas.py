"""Main entry point facade for magma-multigas v2.0."""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .config.logging import get_logger, setup_logging
from .core.exceptions import LoaderError, MagmaMultigasError
from .core.types import DatasetType, DateLike, LogLevel, PathLike
from .data.collection import DatasetCollection
from .data.dataset import Dataset
from .data.loader import DataLoader
from .data.metadata import MetadataExtractor


class MultiGas:
    """Main entry point facade for magma-multigas.

    API improvements over v1.x:
    - Direct property access: mg.six_hours (not .select().get())
    - Configurable logging (no forced print statements)
    - Cached loading for faster re-initialization
    - Type hints throughout

    Example:
        >>> mg = MultiGas(
        ...     six_hours="data/six_hours.dat",
        ...     normalize=True,
        ...     log_level=LogLevel.INFO
        ... )
        >>> data = mg.six_hours
        >>> filtered = data.filter_date_range("2024-05-01", "2024-06-01")
        >>> filtered.save("output/filtered.csv")

    Attributes:
        two_seconds: Two-second interval dataset (if loaded)
        six_hours: Six-hour interval dataset (if loaded)
        one_minute: One-minute interval dataset (if loaded)
        zero: Zero calibration dataset (if loaded)
        span: Span calibration dataset (if loaded)
    """

    def __init__(
        self,
        two_seconds: Optional[PathLike] = None,
        six_hours: Optional[PathLike] = None,
        one_minute: Optional[PathLike] = None,
        zero: Optional[PathLike] = None,
        span: Optional[PathLike] = None,
        normalize: bool = True,
        cache_normalized: bool = True,
        cache_dir: Optional[PathLike] = None,
        log_level: LogLevel = LogLevel.INFO,
    ):
        """Initialize MultiGas with data files.

        Args:
            two_seconds: Path to two-second interval data file
            six_hours: Path to six-hour interval data file
            one_minute: Path to one-minute interval data file
            zero: Path to zero calibration data file
            span: Path to span calibration data file
            normalize: Whether to normalize NAN strings to np.nan
            cache_normalized: Whether to cache normalized files
            cache_dir: Custom cache directory (default: ./output/cache)
            log_level: Logging verbosity level

        Raises:
            MagmaMultigasError: If initialization fails
        """
        # Setup logging
        self.logger = setup_logging(log_level)
        self.logger.info("Initializing MultiGas v2.0")

        # Initialize data loader
        self.loader = DataLoader(cache_dir=Path(cache_dir) if cache_dir else None)
        self.metadata_extractor = MetadataExtractor()

        # Store initialization parameters
        self._normalize = normalize
        self._cache_normalized = cache_normalized

        # Load datasets
        self._datasets: Dict[str, Dataset] = {}

        file_paths = {
            DatasetType.TWO_SECONDS: two_seconds,
            DatasetType.SIX_HOURS: six_hours,
            DatasetType.ONE_MINUTE: one_minute,
            DatasetType.ZERO: zero,
            DatasetType.SPAN: span,
        }

        for dataset_type, file_path in file_paths.items():
            if file_path is not None:
                try:
                    self._load_dataset(dataset_type, file_path)
                except Exception as e:
                    self.logger.error(
                        f"Failed to load {dataset_type.value} from {file_path}: {e}"
                    )
                    # Continue loading other datasets
                    continue

        if not self._datasets:
            self.logger.warning("No datasets were loaded successfully")

        self.logger.info(
            f"MultiGas initialized with {len(self._datasets)} dataset(s): "
            f"{list(self._datasets.keys())}"
        )

    def _load_dataset(self, dataset_type: DatasetType, file_path: PathLike) -> None:
        """Load a single dataset.

        Args:
            dataset_type: Type of dataset
            file_path: Path to data file

        Raises:
            LoaderError: If loading fails
        """
        file_path = Path(file_path)
        self.logger.info(f"Loading {dataset_type.value} from {file_path}")

        # Load data
        df = self.loader.load(
            file_path=file_path,
            dataset_type=dataset_type,
            normalize=self._normalize,
            use_cache=self._cache_normalized,
        )

        # Extract metadata
        metadata = self.metadata_extractor.extract(df, file_path)

        # Also try TOA5 header extraction
        toa5_metadata = self.metadata_extractor.extract_from_toa5_header(file_path)
        if toa5_metadata:
            # Update metadata with TOA5 info
            for key, value in toa5_metadata.items():
                if not hasattr(metadata, key) or getattr(metadata, key) == "Unknown":
                    setattr(metadata, key, value)

        # Create Dataset
        dataset = Dataset(df=df, dataset_type=dataset_type, metadata=metadata)

        # Store dataset
        self._datasets[dataset_type.value] = dataset

        self.logger.info(
            f"Loaded {dataset_type.value}: {len(df)} rows, "
            f"{len(df.columns)} columns, "
            f"station: {metadata.station}"
        )

    @property
    def two_seconds(self) -> Optional[Dataset]:
        """Get two-second interval dataset.

        Returns:
            Dataset or None if not loaded
        """
        return self._datasets.get(DatasetType.TWO_SECONDS.value)

    @property
    def six_hours(self) -> Optional[Dataset]:
        """Get six-hour interval dataset.

        Returns:
            Dataset or None if not loaded
        """
        return self._datasets.get(DatasetType.SIX_HOURS.value)

    @property
    def one_minute(self) -> Optional[Dataset]:
        """Get one-minute interval dataset.

        Returns:
            Dataset or None if not loaded
        """
        return self._datasets.get(DatasetType.ONE_MINUTE.value)

    @property
    def zero(self) -> Optional[Dataset]:
        """Get zero calibration dataset.

        Returns:
            Dataset or None if not loaded
        """
        return self._datasets.get(DatasetType.ZERO.value)

    @property
    def span(self) -> Optional[Dataset]:
        """Get span calibration dataset.

        Returns:
            Dataset or None if not loaded
        """
        return self._datasets.get(DatasetType.SPAN.value)

    def select(self, dataset_type: str) -> Dataset:
        """Select dataset by type name.

        Provided for backward compatibility with v1.x API.
        Prefer direct property access (e.g., mg.six_hours) in v2.0.

        Args:
            dataset_type: Dataset type name (two_seconds, six_hours, etc.)

        Returns:
            Dataset instance

        Raises:
            KeyError: If dataset type doesn't exist
        """
        if dataset_type not in self._datasets:
            available = list(self._datasets.keys())
            raise KeyError(
                f"Dataset type '{dataset_type}' not found. Available types: {available}"
            )
        return self._datasets[dataset_type]

    def as_collection(self) -> DatasetCollection:
        """Get all datasets as a collection.

        Returns:
            DatasetCollection with all loaded datasets
        """
        return DatasetCollection(self._datasets)

    def filter_all(
        self,
        start: Optional[DateLike] = None,
        end: Optional[DateLike] = None,
        inclusive: str = "both",
    ) -> DatasetCollection:
        """Filter all datasets by date range.

        Args:
            start: Start date
            end: End date
            inclusive: Include boundaries ('both', 'left', 'right', 'neither')

        Returns:
            DatasetCollection with filtered datasets
        """
        collection = self.as_collection()
        return collection.filter_all(start, end, inclusive)

    def extract_daily(
        self, start_date: DateLike, end_date: DateLike
    ) -> Dict[str, pd.DataFrame]:
        """Extract daily data for all datasets.

        Convenience method for getting data within date range
        as a dictionary of DataFrames.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary mapping dataset type to filtered DataFrame
        """
        filtered_collection = self.filter_all(start_date, end_date)
        return filtered_collection.to_dict()

    def summary(self) -> pd.DataFrame:
        """Get summary statistics for all loaded datasets.

        Returns:
            DataFrame with summary info for each dataset
        """
        collection = self.as_collection()
        return collection.summary()

    def clear_cache(self) -> int:
        """Clear cached normalized files.

        Returns:
            Number of cache files deleted
        """
        count = self.loader.clear_cache()
        self.logger.info(f"Cleared {count} cache file(s)")
        return count

    def __repr__(self) -> str:
        """String representation of MultiGas instance."""
        dataset_types = list(self._datasets.keys())
        return f"MultiGas(datasets={dataset_types})"
