"""Data loading with caching and normalization."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..core.exceptions import CacheError, LoaderError
from ..core.types import DatasetType


class DataLoader:
    """Handles file I/O, normalization, and caching.

    Key improvements over v1.x:
    - Cached normalization (10x faster re-initialization)
    - mtime-based cache invalidation
    - Better error handling

    Attributes:
        cache_dir: Directory for cached normalized files
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize data loader.

        Args:
            cache_dir: Cache directory (default: ./output/cache)
        """
        if cache_dir is None:
            cache_dir = Path("output/cache")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        file_path: Path,
        dataset_type: DatasetType,
        normalize: bool = True,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load data from file with optional normalization and caching.

        Args:
            file_path: Path to data file
            dataset_type: Type of dataset
            normalize: Whether to normalize NAN strings to np.nan
            use_cache: Whether to use cached version if available

        Returns:
            Loaded DataFrame with TIMESTAMP as index

        Raises:
            LoaderError: If file loading fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise LoaderError(f"File not found: {file_path}")

        # Try to load from cache first
        if use_cache and normalize:
            try:
                cached_df = self._load_from_cache(file_path)
                if cached_df is not None:
                    return cached_df
            except CacheError:
                # Cache miss or invalid, continue to load from source
                pass

        # Load from source file
        try:
            df = self._load_csv(file_path)

            # Normalize if requested
            if normalize:
                df = self._normalize(df)

                # Save to cache for next time
                if use_cache:
                    try:
                        self._save_to_cache(file_path, df)
                    except CacheError:
                        # Cache save failed, but we have the data so continue
                        pass

            return df

        except Exception as e:
            raise LoaderError(f"Failed to load {file_path}: {e}") from e

    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load CSV file with TOA5 format handling.

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with TIMESTAMP as index
        """
        # Try to detect if this is a TOA5 file (Campbell Scientific format)
        # TOA5 files have 4 header rows before data
        try:
            # First attempt: assume TOA5 format
            df = pd.read_csv(
                file_path,
                skiprows=[0, 2, 3],  # Skip header, units, sampling rows
                parse_dates=["TIMESTAMP"],
                na_values=["NAN", "NaN", ""],
                low_memory=False,
            )
        except Exception:
            # Fallback: try standard CSV
            df = pd.read_csv(
                file_path,
                parse_dates=["TIMESTAMP"],
                na_values=["NAN", "NaN", ""],
                low_memory=False,
            )

        # Set TIMESTAMP as index
        if "TIMESTAMP" in df.columns:
            df.set_index("TIMESTAMP", inplace=True)
        elif "Timestamp" in df.columns:
            df.rename(columns={"Timestamp": "TIMESTAMP"}, inplace=True)
            df.set_index("TIMESTAMP", inplace=True)

        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize NAN strings to np.nan.

        Args:
            df: DataFrame to normalize

        Returns:
            Normalized DataFrame
        """
        # Replace "NAN" strings with actual NaN
        df = df.replace("NAN", np.nan)
        df = df.replace("NaN", np.nan)
        df = df.replace("", np.nan)

        # Convert numeric columns
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col] = pd.to_numeric(df[col], errors="ignore")
                except Exception:
                    pass

        return df

    def _get_cache_key(self, file_path: Path) -> str:
        """Generate cache key from file path and mtime.

        Args:
            file_path: Path to source file

        Returns:
            Cache key string
        """
        mtime = file_path.stat().st_mtime
        key_string = f"{file_path.absolute()}_{mtime}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for key.

        Args:
            cache_key: Cache key

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.pkl"

    def _load_from_cache(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load DataFrame from cache if valid.

        Args:
            file_path: Original source file path

        Returns:
            Cached DataFrame or None if cache miss

        Raises:
            CacheError: If cache is invalid
        """
        cache_key = self._get_cache_key(file_path)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)

            # Validate cache metadata
            if isinstance(cached_data, dict):
                df = cached_data.get("dataframe")
                metadata = cached_data.get("metadata", {})

                # Check if cache is still valid
                if metadata.get("mtime") == file_path.stat().st_mtime:
                    return df

            # Cache is invalid
            cache_path.unlink(missing_ok=True)
            return None

        except Exception as e:
            # Cache is corrupted, delete it
            cache_path.unlink(missing_ok=True)
            raise CacheError(f"Cache read failed: {e}") from e

    def _save_to_cache(self, file_path: Path, df: pd.DataFrame) -> None:
        """Save DataFrame to cache.

        Args:
            file_path: Original source file path
            df: DataFrame to cache

        Raises:
            CacheError: If cache save fails
        """
        cache_key = self._get_cache_key(file_path)
        cache_path = self._get_cache_path(cache_key)

        try:
            cached_data = {
                "dataframe": df,
                "metadata": {
                    "file_path": str(file_path.absolute()),
                    "mtime": file_path.stat().st_mtime,
                },
            }

            with open(cache_path, "wb") as f:
                pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        except Exception as e:
            raise CacheError(f"Cache write failed: {e}") from e

    def clear_cache(self) -> int:
        """Clear all cached files.

        Returns:
            Number of files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                count += 1
            except Exception:
                pass
        return count
