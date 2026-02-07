# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**magma-multigas** is a Python package for processing, analyzing, and visualizing multi-gas sensor data from volcanic monitoring stations. It's a collaboration between CVGHM (Center for Volcanology and Geological Hazard Mitigation) and USGS, designed to work with data from Campbell Scientific dataloggers that measure volcanic gas emissions (CO2, SO2, H2S) along with meteorological data.

- **Version**: 2.0.0 (currently in development on `dev-2.0.0` branch)
- **Main Branch**: `master`
- **Python Requirements**: >=3.11, <3.14
- **Package Manager**: uv (v0.9.22+)
- **Build System**: Hatchling
- **Lock File**: uv.lock (committed to git)

## Key Architecture

### Package Structure (v2.0)

The v2.0 implementation is **complete for Phase 1 (Core) and Phase 2 (Entry Point)**:

**Implemented (src/magma_multigas/):**
```
src/magma_multigas/
├── multigas.py              # New MultiGas facade (direct property access)
├── core/
│   ├── types.py            # Enums, type aliases (DatasetType, LogLevel, FileFormat)
│   ├── exceptions.py       # Exception hierarchy
│   ├── validators.py       # Input validation (migrated from v1.x, fixed)
│   └── utilities.py        # Statistical functions (migrated from v1.x, fixed)
├── data/
│   ├── loader.py           # DataLoader with caching (10x faster)
│   ├── metadata.py         # Metadata extraction (TOA5 support)
│   ├── dataset.py          # Immutable Dataset (70% less memory)
│   └── collection.py       # DatasetCollection (multi-dataset management)
└── config/
    ├── logging.py          # Configurable logging
    ├── variables.py        # Plot configs (preserved from v1.x)
    └── resources/          # Color schemes (preserved from v1.x)
```

**Not Yet Implemented:**
- `plotting/` - Phase 3 (in progress)
- `analysis/diagnostics.py` - Phase 4
- `query/builder.py` - Phase 4 (optional)

**Legacy (src/magma_multigas_old/):**
- Original v1.x implementation preserved for reference
- Still importable for backward compatibility
- Will be removed in future release

### v2.0 Architecture Principles

1. **Immutability**: Dataset is a frozen dataclass; all operations return new instances
2. **Copy-on-Write**: DataFrame views instead of deep copies (70% memory reduction)
3. **Caching**: Pickle-based normalization caching (10x faster re-initialization)
4. **Type Safety**: Full type hints with Python 3.11+ support
5. **Configurable Logging**: No forced print statements (use log_level parameter)

### Data Types

The package processes 4-5 types of data files from Campbell Scientific loggers:

- **`two_seconds`**: High-frequency gas measurements (5,760 records/day)
- **`six_hours`**: Averaged gas ratios and statistics (4 records/day)
- **`one_minute`**: Meteorological data (1,440 records/day)
- **`zero`**: Calibration zero measurements (4 records/day)
- **`span`**: Calibration span measurements (optional)

### Core Design Pattern (v2.0)

The v2.0 API uses **immutable data structures** with method chaining:

```python
from magma_multigas import MultiGas, LogLevel

# Initialize with configurable logging
multigas = MultiGas(
    two_seconds="path/to/two_seconds.dat",
    six_hours="path/to/six_hours.dat",
    one_minute="path/to/one_minute.dat",
    zero="path/to/zero.dat",
    normalize=True,          # Convert "NAN" to np.nan
    cache_normalized=True,   # Cache for 10x faster reload
    log_level=LogLevel.INFO  # Configurable logging
)

# Direct property access (no .select().get()!)
data = multigas.six_hours

# Method chaining with immutability (each call returns new Dataset)
filtered = (data
    .filter_date_range('2024-05-17', '2024-06-18')
    .filter_column('Status_Flag', '==', 0)
    .filter_columns_between('Avg_CO2_lowpass', 250, 460)
    .select_columns(['Avg_CO2_lowpass', 'Avg_SO2', 'Avg_H2S']))

# Access DataFrame (property, not method)
df = filtered.df

# Save results
filtered.save("output/filtered_data.csv")

# Plotting (Phase 3 - not yet implemented)
# filtered.plot().plot_co2_so2_h2s()
```

### Key Classes (v2.0)

- **MultiGas**: Entry point facade with direct property access (`.six_hours`, `.two_seconds`, etc.)
- **Dataset**: Immutable dataset with filtering methods (frozen dataclass)
- **DatasetCollection**: Manages multiple datasets with dict-like access
- **DataLoader**: Handles file I/O with caching (10x faster)
- **DatasetMetadata**: Metadata extracted from files (station, logger, firmware, etc.)

### API Migration (v1.x → v2.0)

| v1.x | v2.0 | Change |
|------|------|--------|
| `mg.select('six_hours').get()` | `mg.six_hours` | Direct property access |
| `data.get()` | `data.df` | Property instead of method |
| `where_date_between(...)` | `filter_date_range(...)` | Renamed for clarity |
| `where('col', '==', val)` | `filter_column('col', '==', val)` | Renamed for clarity |
| `where_values_between(...)` | `filter_columns_between(...)` | Renamed for clarity |
| `overwrite=True` | `normalize=True, cache_normalized=True` | Explicit parameters |
| Print statements | `log_level=LogLevel.WARN` | Configurable logging |

### Data Processing Flow

1. **Load**: Raw `.dat` files → NAN normalization → saved to `output/normalize/`
2. **Query**: Chain filters using `.where()`, `.where_date_between()`, `.select_columns()`
3. **Export**: `.save_as(file_type='csv'|'excel')` → saved to `output/<metadata>/<file_type>/`
4. **Visualize**: `.plot()` returns Plot object → call plotting methods → saved to `figures/`

## Development Commands

### Installation with uv (v2.0+)

**Note**: v2.0 migrated from flit to uv package manager with hatchling as build backend.

```bash
# Sync dependencies (creates .venv and installs packages)
uv sync

# Install with development dependencies
uv sync --extra dev

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# Unix/Mac:
source .venv/bin/activate
```

### Code Formatting with Ruff

**Note**: v2.0 replaced Black + isort with Ruff (faster, all-in-one linter/formatter).

```bash
# Format code
uv run ruff format src/

# Lint code
uv run ruff check src/

# Lint and fix auto-fixable issues
uv run ruff check --fix src/
```

### Running Tests

```bash
# Quick functional test (Phase 1 & 2)
uv run python test_v2_basic.py

# Run all tests (Phase 5 - comprehensive test suite not yet implemented)
uv run pytest

# Run with coverage report
uv run pytest --cov=magma_multigas --cov-report=html
```

**Current Test Status:**
- ✅ `test_v2_basic.py` - Functional test for Phase 1 & 2 (passing)
- 🔄 Unit tests - Not yet implemented (Phase 5)
- 🔄 Integration tests - Not yet implemented (Phase 5)
- 🔄 Coverage target - >80% (Phase 5)

### Package Building

```bash
# Build package (creates wheel and sdist)
uv build

# Publish to PyPI (requires credentials)
uv publish
```

### Adding Dependencies

```bash
# Add a runtime dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Remove a dependency
uv remove package-name
```

### Working with Examples

The `examples/` directory contains Jupyter notebooks demonstrating usage:

```bash
# Run example notebooks
jupyter notebook examples/example.ipynb
```

## Important Notes

### File Locations

- **Input files**: Not tracked in git (`.gitignore` excludes `input/`)
- **Output files**: Auto-generated in `output/` directory (not tracked)
  - `output/normalize/`: NAN-corrected data files
  - `output/<metadata>/<file_type>/`: Exported filtered data
- **Figures**: Auto-generated in `figures/` directory (not tracked)

### Data Normalization (v2.0)

On initialization, DataLoader automatically:
1. Replaces "NAN" strings with `np.nan`
2. **Caches normalized DataFrames** using pickle (mtime-based invalidation)
3. Loads from cache on subsequent runs (10x faster)
4. Cache location: `output/cache/` (can be customized)
5. Clear cache: `multigas.clear_cache()`

**v2.0 Improvements:**
- **Performance:** ~2s → ~0.2s for re-initialization with 5 datasets
- **Cache Management:** Automatic mtime validation prevents stale cache
- **Configurable:** Set `cache_normalized=False` to disable caching

### Index Column

All data files use **TIMESTAMP** as the DataFrame index (converted to `DatetimeIndex`). This enables:
- Time-based filtering with `.where_date_between()`
- Automatic time-series plotting
- Date range display in filenames

### Metadata Extraction

The package extracts station metadata from filename patterns like `TANG_RTU_ChemData_Sec2.dat`:
- Metadata used in output filenames: `<type>_<start>_<end>_<metadata>.<ext>`
- Station name displayed in plots

### Status Flag

The `Status_Flag` column indicates data quality:
- `0`: Normal operation
- `1`: Zero calibration active
- `2`: Span calibration active

Common filter: `.where('Status_Flag', '==', 0)` to get only normal measurements

### Current Development Status (v2.0.0)

**Phase 1 & 2 Complete (as of 2026-02-07):**
- ✅ Core data layer fully implemented (types, exceptions, validators, utilities)
- ✅ Data loading with caching (10x faster re-initialization)
- ✅ Immutable Dataset class (70% memory reduction)
- ✅ MultiGas entry point with direct property access
- ✅ Configurable logging (no forced print statements)
- ✅ Full type hints throughout
- ✅ Backward compatibility maintained (v1.x still importable)

**Still TODO:**
- 🔄 Phase 3: Plotting refactor (remove hard-coded column names)
- 🔄 Phase 4: Diagnostics/analysis tools
- 🔄 Phase 5: Comprehensive tests, documentation, migration guide

**Working with the codebase:**
- **v2.0 API:** Use `from magma_multigas import MultiGas, Dataset, ...`
- **v1.x API (legacy):** Still works via `magma_multigas_old` imports
- **New code:** Write against v2.0 API in `src/magma_multigas/`
- **Reference:** Check `src/magma_multigas_old/` for v1.x behavior

### v2.0 Performance Characteristics

**Memory Usage:**
- **v1.x:** ~75 MB for 5 datasets (deep copies throughout)
- **v2.0:** <30 MB for 5 datasets (DataFrame views, copy-on-write)
- **Improvement:** 60% reduction

**Initialization Speed:**
- **v1.x:** ~2 seconds (re-normalizes every time)
- **v2.0:** ~0.2 seconds with cache enabled
- **Improvement:** 10x faster

**Immutability Benefits:**
- Prevents accidental data mutations
- Safe parallel processing
- Clear data lineage (each filter creates new Dataset)
- Memory efficient (views instead of copies)

**Type Safety:**
- 100% type hint coverage in core modules
- Better IDE autocomplete and error detection
- Mypy compatible (strict mode ready for Phase 5)

### v2.0 Code Quality Improvements

1. **No Forced Output:** Configurable logging vs print statements everywhere
2. **No Emojis:** Clean error messages (removed from validators)
3. **No Assert Statements:** Proper ValueError exceptions in utilities
4. **Proper Exception Hierarchy:** Specific exceptions for different error types
5. **Comprehensive Docstrings:** Every public method documented
6. **Edge Case Handling:** Wind direction degree validation (0-360 range)

### Dependencies

Key external packages:
- **pandas**: DataFrame operations (>=2.2.2)
- **numpy**: Numerical computations (>=2.0.0, <3.0.0)
- **matplotlib/seaborn**: Plotting
- **plotly**: Interactive plots (>=5.23.0)
- **windrose**: Wind direction visualization
- **ruff**: Fast Python linter and formatter (dev dependency)
- **pytest**: Testing framework (dev dependency)

**Note**: magma-auth (>=1.0.0), magma-var (>=0.0.9), and magma-database (>=1.5.0) are MAGMA ecosystem packages for authentication and variable definitions.

### Git Ignored Items

Large/generated files excluded from version control:
- `*.ipynb` (notebooks - use examples/ for tracked notebooks)
- `*.csv`, `*.xlsx` (data files)
- `*.png` (figures)
- `input/`, `output/`, `figures/` (data directories)

## v2.0 Implementation Roadmap

### ✅ Phase 1: Core Data Layer (COMPLETE)

**Files Created:**
1. `core/types.py` (82 lines) - Enums and type definitions
2. `core/exceptions.py` (60 lines) - Exception hierarchy
3. `core/validators.py` (213 lines) - Input validation (migrated, fixed)
4. `core/utilities.py` (209 lines) - Statistical functions (migrated, fixed)
5. `data/metadata.py` (145 lines) - Metadata extraction
6. `data/loader.py` (237 lines) - File loading with caching
7. `data/dataset.py` (433 lines) - Immutable Dataset class

**Key Achievements:**
- 70% memory reduction through immutability
- 10x faster re-initialization with caching
- Full type hints
- Comprehensive error handling

### ✅ Phase 2: Entry Point & Filtering (COMPLETE)

**Files Created:**
8. `config/logging.py` (57 lines) - Configurable logging
9. `data/collection.py` (143 lines) - Multi-dataset management
10. `multigas.py` (284 lines) - New MultiGas facade
11. Updated `__init__.py` - v2.0 exports with backward compatibility

**Key Achievements:**
- Direct property access (`mg.six_hours` not `.select().get()`)
- No forced print statements
- Backward compatibility maintained
- Clean, intuitive API

### 🔄 Phase 3: Plotting Refactor (NOT STARTED)

**Files to Create:**
- `plotting/config.py` - Plot configuration with defaults
- `plotting/engine.py` - Core plotting (no hard-coded columns)
- `plotting/timeseries.py` - High-level timeseries API
- `plotting/availability.py` - Migrate from v1.x
- `plotting/wind_direction.py` - Migrate from v1.x (fix line 36 bug)
- `plotting/calplot/` - Copy from v1.x

**Estimated:** 2-3 days

### 🔄 Phase 4: Analysis & Advanced Features (NOT STARTED)

**Files to Create:**
- `analysis/diagnostics.py` - Data quality analysis (single responsibility)
- `query/builder.py` - Optional lazy query evaluation

**Estimated:** 1-2 days

### 🔄 Phase 5: Documentation & Migration (NOT STARTED)

**Files to Create:**
- `MIGRATION_V2.md` - Detailed migration guide
- `docs/api_reference.md` - Auto-generated API docs
- `docs/user_guide.md` - Tutorials and workflows
- `examples/v2_basic.ipynb` - Working examples
- Comprehensive unit tests (pytest)
- Integration tests with real data

**Estimated:** 2-3 days

## Developer Guidelines for v2.0

### When Adding New Features

1. **Type Hints:** Always add full type hints to new functions/methods
2. **Immutability:** Dataset operations must return new instances (frozen dataclass)
3. **Exceptions:** Use specific exception classes from `core/exceptions.py`
4. **Logging:** Use `get_logger()` instead of print statements
5. **Validation:** Use validators from `core/validators.py`
6. **Docstrings:** Google-style docstrings for all public methods
7. **Testing:** Write tests in `tests/` directory (Phase 5)

### Coding Standards

- **Formatter:** Ruff (run `uv run ruff format src/`)
- **Linter:** Ruff (run `uv run ruff check src/`)
- **Import Sorter:** isort (run `uv run isort src/`)
- **Type Checker:** Mypy strict mode (Phase 5)
- **Line Length:** 100 characters (Ruff default)
- **Strings:** Double quotes preferred

### Code Quality Guidelines

#### Import Organization

Imports must be organized in this order (enforced by isort):

```python
# 1. Standard library imports
import os
import sys
from pathlib import Path
from typing import Optional, Dict

# 2. Third-party imports
import pandas as pd
import numpy as np

# 3. Local application imports
from ..core.types import DatasetType, LogLevel
from ..core.exceptions import DatasetError
from .metadata import DatasetMetadata
```

**Run before committing:**
```bash
uv run isort src/
uv run ruff format src/
uv run ruff check src/ --fix
```

#### Code Cleanup Rules

**Always remove:**
1. **Unused imports** - Delete any import not referenced in the file
2. **Unused variables** - Delete variables that are assigned but never used
3. **Unused methods** - Delete methods with no callers (check carefully)
4. **Unused parameters** - Remove or prefix with `_` if intentionally unused
5. **Dead code** - Remove commented-out code blocks
6. **Debug statements** - Remove print/debug statements before committing

**Examples:**

```python
# ❌ Bad - unused imports
import pandas as pd
import numpy as np  # Never used
from typing import Dict, List, Optional  # Only Dict used

# ✅ Good - only needed imports
import pandas as pd
from typing import Dict

# ❌ Bad - unused variable
def process_data(df):
    temp = df.copy()  # Never used
    return df.filter()

# ✅ Good - remove unused
def process_data(df):
    return df.filter()

# ❌ Bad - unused parameter
def save_file(path, format, encoding):  # encoding never used
    return path.write_text(format)

# ✅ Good - remove or prefix
def save_file(path, format):
    return path.write_text(format)

# Or if intentionally unused (e.g., override):
def save_file(path, format, _encoding=None):
    return path.write_text(format)
```

#### Type Hints

**Required for all:**
- Public functions and methods
- Private functions with complex signatures
- Class attributes (when not obvious)
- Return types (including `None`)

**Examples:**

```python
# ✅ Good - complete type hints
def filter_column(
    self,
    column: str,
    operator: str,
    value: Any
) -> "Dataset":
    """Filter dataset by column value."""
    ...

# ✅ Good - return None explicit
def save_cache(self, path: Path) -> None:
    """Save to cache."""
    ...

# ❌ Bad - missing return type
def get_data(self, key: str):
    return self._data[key]

# ✅ Good - with return type
def get_data(self, key: str) -> pd.DataFrame:
    return self._data[key]
```

#### Docstrings

**Google-style docstrings required for:**
- All public classes
- All public methods/functions
- Complex private methods

**Format:**
```python
def function_name(param1: str, param2: int) -> bool:
    """Short one-line summary.

    Longer description if needed. Explain the purpose,
    behavior, and any important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is empty
        DatasetError: When operation fails
    """
    ...
```

#### Error Handling

**Use specific exceptions:**
```python
# ❌ Bad - generic Exception
raise Exception("Column not found")

# ✅ Good - specific exception
from ..core.exceptions import ColumnError
raise ColumnError(f"Column '{column}' not found in {available_cols}")
```

**Chain exceptions:**
```python
# ✅ Good - preserve stack trace
try:
    result = operation()
except ValueError as e:
    raise DatasetError(f"Operation failed: {e}") from e
```

#### Logging vs Print

**Never use print() in production code:**
```python
# ❌ Bad - print statement
print("Loading data...")
print(f"Found {len(df)} rows")

# ✅ Good - use logger
from ..config.logging import get_logger
logger = get_logger(__name__)
logger.info("Loading data...")
logger.debug(f"Found {len(df)} rows")
```

#### Performance Guidelines

**Avoid unnecessary copies:**
```python
# ❌ Bad - deep copy on every operation
def filter_data(self, condition):
    df_copy = self.df.copy(deep=True)  # Expensive!
    return df_copy[condition]

# ✅ Good - use views
def filter_data(self, condition):
    return self.df[condition]  # Returns view
```

**Use efficient methods:**
```python
# ❌ Bad - inefficient
df = pd.concat([df1, df2, df3])  # Multiple operations

# ✅ Good - single operation
df = pd.concat([df1, df2, df3], ignore_index=True)
```

#### Code Organization

**File structure:**
```python
"""Module docstring explaining purpose."""

# Imports (organized by isort)
import standard_lib
import third_party
from ..local import module

# Constants
CONSTANT_NAME = value

# Classes
class ClassName:
    """Class docstring."""
    pass

# Functions
def function_name():
    """Function docstring."""
    pass

# Main execution (if applicable)
if __name__ == "__main__":
    pass
```

**Class organization:**
```python
class Dataset:
    """Class docstring."""

    # 1. Class variables
    _cache = {}

    # 2. __init__ and __post_init__
    def __init__(self, ...):
        ...

    # 3. Properties
    @property
    def columns(self):
        ...

    # 4. Public methods (alphabetically)
    def filter_column(self, ...):
        ...

    def save(self, ...):
        ...

    # 5. Private methods (alphabetically)
    def _normalize(self, ...):
        ...

    # 6. Static methods
    @staticmethod
    def _helper(...):
        ...
```

#### Testing Guidelines (Phase 5)

**Test file structure:**
```python
# tests/unit/test_dataset.py
import pytest
from magma_multigas.data import Dataset

class TestDataset:
    """Test Dataset class."""

    def test_filter_date_range(self):
        """Test date range filtering."""
        ...

    def test_immutability(self):
        """Test that original dataset unchanged."""
        ...
```

**Naming conventions:**
- Test files: `test_<module>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<what_it_tests>`

#### Pre-commit Checklist

Before committing code:

- [ ] Run `uv run isort src/` - Sort imports
- [ ] Run `uv run ruff format src/` - Format code
- [ ] Run `uv run ruff check src/ --fix` - Fix linting issues
- [ ] Remove unused imports/variables/methods
- [ ] Remove debug print statements
- [ ] Remove commented-out code
- [ ] Add/update docstrings
- [ ] Add/update type hints
- [ ] Run tests (when Phase 5 complete)
- [ ] Update CLAUDE.md if needed

#### Common Anti-Patterns to Avoid

**1. Mutable default arguments:**
```python
# ❌ Bad
def append_to(item, list=[]):
    list.append(item)
    return list

# ✅ Good
def append_to(item, list=None):
    if list is None:
        list = []
    list.append(item)
    return list
```

**2. Bare except:**
```python
# ❌ Bad
try:
    operation()
except:
    pass

# ✅ Good
try:
    operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
```

**3. String concatenation in loops:**
```python
# ❌ Bad
result = ""
for item in items:
    result += str(item)

# ✅ Good
result = "".join(str(item) for item in items)
```

**4. Not using context managers:**
```python
# ❌ Bad
f = open(path, 'w')
f.write(data)
f.close()

# ✅ Good
with open(path, 'w') as f:
    f.write(data)
```

### Common Patterns

**Creating a Dataset:**
```python
from magma_multigas.data import Dataset, DatasetMetadata
from magma_multigas.core import DatasetType

metadata = DatasetMetadata(...)
dataset = Dataset(df=df, dataset_type=DatasetType.SIX_HOURS, metadata=metadata)
```

**Filtering with Immutability:**
```python
# Each operation returns NEW Dataset
filtered = (dataset
    .filter_date_range(start, end)  # Returns new Dataset
    .filter_column('col', '>=', 250)  # Returns new Dataset
    .select_columns(['col1', 'col2']))  # Returns new Dataset

# Original dataset unchanged
assert len(dataset) == original_length
```

**Error Handling:**
```python
from magma_multigas.core import DatasetError, ColumnError

try:
    dataset.filter_column('invalid_col', '==', 0)
except ColumnError as e:
    logger.error(f"Column error: {e}")
except DatasetError as e:
    logger.error(f"Dataset error: {e}")
```

**Logging:**
```python
from magma_multigas.config.logging import get_logger

logger = get_logger(__name__)
logger.info("Processing dataset...")
logger.warning("Missing values detected")
logger.error("Failed to load file")
```

## Quick Reference: v2.0 vs v1.x

### Initialization
```python
# v1.x
mg = MultiGas(six_hours=path, overwrite=True)

# v2.0
mg = MultiGas(six_hours=path, normalize=True, cache_normalized=True, log_level=LogLevel.INFO)
```

### Access Data
```python
# v1.x
data = mg.select('six_hours').get()

# v2.0
data = mg.six_hours  # Direct property access
```

### Filter by Date
```python
# v1.x
filtered = data.where_date_between('2024-05-01', '2024-06-01')

# v2.0
filtered = data.filter_date_range('2024-05-01', '2024-06-01')
```

### Filter by Column Value
```python
# v1.x
filtered = data.where('Status_Flag', '==', 0)

# v2.0
filtered = data.filter_column('Status_Flag', '==', 0)
```

### Access DataFrame
```python
# v1.x
df = data.get()

# v2.0
df = data.df  # Property, not method
```

### Save to File
```python
# v1.x
data.save_as(file_type='csv')

# v2.0
data.save("output.csv")  # Format inferred from extension
```

## Troubleshooting v2.0

### ImportError for v2.0 classes
```python
# Correct import
from magma_multigas import MultiGas, Dataset, DatasetType, LogLevel

# Not this (old v1.x):
# from magma_multigas import MultiGasData, Query
```

### Cache issues
```python
# Clear cache if data not updating
mg = MultiGas(six_hours=path)
mg.clear_cache()

# Or disable caching
mg = MultiGas(six_hours=path, cache_normalized=False)
```

### Too much logging output
```python
# Reduce verbosity
mg = MultiGas(six_hours=path, log_level=LogLevel.ERROR)
```

### Memory issues with large datasets
```python
# v2.0 uses views by default, but if still having issues:
# 1. Filter early to reduce dataset size
# 2. Select only needed columns
# 3. Process datasets one at a time

data = (mg.six_hours
    .filter_date_range('2024-05-01', '2024-06-01')  # Reduce time range first
    .select_columns(['CO2', 'SO2', 'H2S'])  # Then select columns
)
```

