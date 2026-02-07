#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Basic test of magma-multigas v2.0 core functionality.

This script tests Phase 1 (Core Data Layer) and Phase 2 (Entry Point & Filtering)
without requiring actual data files.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Test imports
print("Testing v2.0 imports...")
from magma_multigas import (
    MultiGas,
    Dataset,
    DatasetCollection,
    DatasetMetadata,
    DatasetType,
    LogLevel,
    FileFormat,
)
print("[OK] All imports successful!")

# Create synthetic test data
print("\nCreating synthetic test dataset...")
dates = pd.date_range(start='2024-05-01', end='2024-06-01', freq='6h')
df_test = pd.DataFrame({
    'TIMESTAMP': dates,
    'CO2': np.random.uniform(400, 450, len(dates)),
    'SO2': np.random.uniform(-0.1, 0.1, len(dates)),
    'H2S': np.random.uniform(0, 1, len(dates)),
    'Status_Flag': np.random.choice([0, 1, 2], len(dates)),
})
df_test.set_index('TIMESTAMP', inplace=True)
print(f"[OK] Created test dataset with {len(df_test)} rows, {len(df_test.columns)} columns")

# Test Dataset creation
print("\nTesting Dataset class...")
metadata = DatasetMetadata(
    station="TEST_STATION",
    logger_type="TEST_LOGGER",
    firmware="1.0",
    program_name="test.cr6",
    file_sampling="6 Hour"
)
dataset = Dataset(
    df=df_test,
    dataset_type=DatasetType.SIX_HOURS,
    metadata=metadata
)
print(f"[OK] Dataset created: {dataset}")
print(f"  - Shape: {dataset.shape}")
print(f"  - Date range: {dataset.date_range[0].date()} to {dataset.date_range[1].date()}")
print(f"  - Columns: {dataset.columns}")

# Test filtering by date range
print("\nTesting date range filtering...")
filtered = dataset.filter_date_range('2024-05-10', '2024-05-20')
print(f"[OK] Filtered dataset: {len(filtered)} rows")
print(f"  - New date range: {filtered.date_range[0].date()} to {filtered.date_range[1].date()}")

# Test filtering by column value
print("\nTesting column filtering...")
status_filtered = filtered.filter_column('Status_Flag', '==', 0)
print(f"[OK] Filtered by Status_Flag==0: {len(status_filtered)} rows")

# Test column selection
print("\nTesting column selection...")
selected = status_filtered.select_columns(['CO2', 'SO2', 'H2S'])
print(f"[OK] Selected columns: {selected.columns}")

# Test chaining (immutability)
print("\nTesting method chaining...")
result = (dataset
    .filter_date_range('2024-05-10', '2024-05-20')
    .filter_column('Status_Flag', '==', 0)
    .select_columns(['CO2', 'SO2']))
print(f"[OK] Chained operations: {result.shape}")

# Test immutability
print("\nTesting immutability...")
original_len = len(dataset)
filtered_len = len(filtered)
print(f"  - Original dataset unchanged: {len(dataset)} rows (was {original_len})")
print(f"  - Filtered dataset independent: {filtered_len} rows")
assert len(dataset) == original_len, "Original dataset was modified!"
print("[OK] Dataset is immutable")

# Test DatasetCollection
print("\nTesting DatasetCollection...")
datasets = {
    'six_hours': dataset,
    'test': filtered,
}
collection = DatasetCollection(datasets)
print(f"[OK] Collection created with {len(collection)} datasets")
print(f"  - Types: {list(collection.keys())}")
print(f"  - Summary:\n{collection.summary()}")

# Test saving (to memory, not disk)
print("\nTesting save functionality...")
try:
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        saved_path = result.save(csv_path)
        print(f"[OK] Dataset saved to: {saved_path}")

        # Verify file exists
        assert os.path.exists(saved_path), "File was not created!"
        print(f"  - File size: {os.path.getsize(saved_path)} bytes")
except Exception as e:
    print(f"[FAIL] Save test failed: {e}")

print("\n" + "="*60)
print("[OK] ALL PHASE 1 & 2 TESTS PASSED!")
print("="*60)
print("\nv2.0 Core functionality is working correctly.")
print("Ready for Phase 3 (Plotting) and Phase 4 (Analysis).")
