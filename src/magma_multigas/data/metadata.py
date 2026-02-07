"""Metadata extraction and management for datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ..core.exceptions import MetadataError


@dataclass
class DatasetMetadata:
    """Metadata extracted from CSV headers and filenames.

    Attributes:
        station: Station name (e.g., 'TANG_RTU')
        logger_type: Logger type (e.g., 'CR6')
        firmware: Firmware version (e.g., 'CR6.10')
        program_name: Program name (e.g., 'CPU:ChemData_Sec2.CR6')
        file_sampling: Sampling interval (e.g., '2 Sec', '6 Hour')
        serial_number: Logger serial number (optional)
        os_version: Operating system version (optional)
    """

    station: str
    logger_type: str
    firmware: str
    program_name: str
    file_sampling: str
    serial_number: Optional[str] = None
    os_version: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetMetadata":
        """Create metadata from dictionary.

        Args:
            data: Dictionary with metadata fields

        Returns:
            DatasetMetadata instance
        """
        return cls(
            station=data.get("station", "Unknown"),
            logger_type=data.get("logger_type", "Unknown"),
            firmware=data.get("firmware", "Unknown"),
            program_name=data.get("program_name", "Unknown"),
            file_sampling=data.get("file_sampling", "Unknown"),
            serial_number=data.get("serial_number"),
            os_version=data.get("os_version"),
        )


class MetadataExtractor:
    """Extract and validate metadata from data files."""

    def extract(self, df: pd.DataFrame, file_path: Path) -> DatasetMetadata:
        """Extract metadata from DataFrame and filename.

        Metadata is extracted from:
        1. CSV header rows (TOA5 format from Campbell Scientific)
        2. Filename pattern (e.g., TANG_RTU_ChemData_Sec2.dat)

        Args:
            df: DataFrame loaded from CSV
            file_path: Path to the data file

        Returns:
            DatasetMetadata instance

        Raises:
            MetadataError: If metadata extraction fails
        """
        try:
            metadata = {}

            # Extract from filename
            filename_metadata = self._extract_from_filename(file_path)
            metadata.update(filename_metadata)

            # Extract from DataFrame columns if available
            if hasattr(df, "attrs") and df.attrs:
                metadata.update(df.attrs)

            # Set defaults for missing fields
            metadata.setdefault("station", "Unknown")
            metadata.setdefault("logger_type", "Unknown")
            metadata.setdefault("firmware", "Unknown")
            metadata.setdefault("program_name", "Unknown")
            metadata.setdefault("file_sampling", "Unknown")

            return DatasetMetadata.from_dict(metadata)

        except Exception as e:
            raise MetadataError(
                f"Failed to extract metadata from {file_path}: {e}"
            ) from e

    def _extract_from_filename(self, file_path: Path) -> dict:
        """Extract metadata from filename pattern.

        Expected patterns:
        - STATION_DEVICE_DataType_Interval.dat
        - e.g., TANG_RTU_ChemData_Sec2.dat → station=TANG_RTU

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with extracted metadata
        """
        filename = file_path.stem  # Get filename without extension
        parts = filename.split("_")

        metadata = {}

        if len(parts) >= 2:
            # First part is typically station name
            metadata["station"] = parts[0]

            # Try to infer logger type
            if len(parts) >= 2 and parts[1] in ("RTU", "LOGGER", "CR1000", "CR6"):
                metadata["logger_type"] = parts[1]

        return metadata

    def extract_from_toa5_header(self, file_path: Path) -> dict:
        """Extract metadata from TOA5 CSV header.

        TOA5 format (Campbell Scientific):
        Line 1: "TOA5", station_name, model, serial_no, os_version, program, signature, table_name
        Line 2: TIMESTAMP, column1, column2, ...
        Line 3: TS, units, units, ...
        Line 4: , Smp, Smp, ...

        Args:
            file_path: Path to TOA5 CSV file

        Returns:
            Dictionary with header metadata
        """
        metadata = {}

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()

                # Check if TOA5 format
                if not first_line.startswith('"TOA5"'):
                    return metadata

                # Parse header fields
                parts = [p.strip('"') for p in first_line.split(",")]

                if len(parts) >= 8:
                    metadata["logger_type"] = "TOA5"
                    metadata["station"] = parts[1]
                    metadata["logger_model"] = parts[2]
                    metadata["serial_number"] = parts[3]
                    metadata["os_version"] = parts[4]
                    metadata["program_name"] = parts[5]
                    metadata["file_sampling"] = parts[7]

        except Exception:
            # If header parsing fails, return empty dict
            pass

        return metadata
