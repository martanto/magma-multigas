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

### Package Structure

The project is undergoing a major refactoring:

- **`src/magma_multigas/`**: New v2.0 implementation (currently minimal - contains only placeholder files)
- **`src/magma_multigas_old/`**: Original v1.x implementation (fully functional)

The old implementation uses a class-based architecture with method chaining for data queries:

1. **`multigas.py`**: Main entry point - `MultiGas` class that loads and manages multiple data files
2. **`multigas_data.py`**: Core data handling - `MultiGasData` class extends `Query` for fluent filtering
3. **`query.py`**: Query engine - `Query` base class provides chainable filter methods
4. **`plot.py`**: Visualization - `Plot` class for creating various plots using matplotlib/seaborn
5. **`diagnose.py`**: Data quality - `Diagnose` class for checking completeness and missing data
6. **`validator.py`**: Input validation utilities
7. **`utilities.py`**: Helper functions for data transformations
8. **`variables.py`**: Configuration and plot properties

### Data Types

The package processes 4-5 types of data files from Campbell Scientific loggers:

- **`two_seconds`**: High-frequency gas measurements (5,760 records/day)
- **`six_hours`**: Averaged gas ratios and statistics (4 records/day)
- **`one_minute`**: Meteorological data (1,440 records/day)
- **`zero`**: Calibration zero measurements (4 records/day)
- **`span`**: Calibration span measurements (optional)

### Core Design Pattern

The package uses **fluent interface pattern** with method chaining:

```python
multigas = MultiGas(two_seconds=..., six_hours=..., one_minute=..., zero=...)

# Select and filter data with chaining
filtered = (multigas.select('two_seconds').get()
    .select_columns(['H2O','CO2','SO2','H2S'])
    .where_date_between('2024-05-17', '2024-06-18')
    .where('Status_Flag', '==', 0)
    .where_values_between('SO2', -0.129, -0.127))

# Plot results
filtered.plot().plot_co2_so2_h2s()
```

### Key Classes

- **MultiGas**: Container for all data types, provides `.select(type)` method
- **MultiGasData**: Extends `Query`, represents a single data type with filtering/export capabilities
- **Query**: Base class with chainable methods (`.where()`, `.where_date_between()`, `.select_columns()`)
- **Plot**: Matplotlib-based plotting with preset methods for common visualizations
- **Diagnose**: Data quality analysis (completeness, missing values, availability)

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
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=magma_multigas --cov-report=html
```

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

### Data Normalization

On initialization, `MultiGasData` automatically:
1. Replaces "NAN" strings with `np.nan`
2. Saves normalized files to `output/normalize/`
3. Uses normalized files for all subsequent operations
4. Set `overwrite=True` to force re-normalization

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

The repository is in transition:
- All functional code is in `src/magma_multigas_old/`
- The new `src/magma_multigas/magma_multigas.py` is nearly empty
- `__init__.py` still imports from old module structure
- When making changes, work with the `_old` directory until migration is complete

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

