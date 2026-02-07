# magma-multigas

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**High-performance Python package for processing, analyzing, and visualizing multi-gas volcanic monitoring data.**

Developed through collaboration between CVGHM (Center for Volcanology and Geological Hazard Mitigation) and USGS for Campbell Scientific datalogger systems measuring volcanic gas emissions (CO2, SO2, H2S) and meteorological data.

---

## ✨ What's New in v2.0

- 🚀 **136x faster** with intelligent caching
- 💾 **67% less memory** usage
- 🔒 **Immutable data structures** prevent accidental modifications
- 🎯 **Cleaner API** - no more redundant `.get()` calls
- 📝 **Full type hints** for better IDE support
- ⚙️ **Configurable logging** - control verbosity
- ✅ **Production tested** with real volcanic monitoring data

---

## 📦 Installation

### Requirements
- Python 3.11 or higher
- Windows, macOS, or Linux

### Install with pip

```bash
pip install magma-multigas
```

### Install with uv (recommended for development)

```bash
# Install uv package manager
pip install uv

# Clone repository
git clone https://github.com/martanto/magma-multigas.git
cd magma-multigas

# Install with dependencies
uv sync
```

---

## 🚀 Quick Start

### Basic Usage

```python
from magma_multigas import MultiGas, LogLevel

# Initialize with data files
multigas = MultiGas(
    two_seconds='data/TANG_RTU_ChemData_Sec2.dat',
    six_hours='data/TANG_RTU_Data_6Hr.dat',
    one_minute='data/TANG_RTU_Wx_Min1.dat',
    zero='data/TANG_RTU_Zero_Data.dat',
    normalize=True,           # Convert "NAN" strings to np.nan
    cache_normalized=True,    # Enable caching (136x faster!)
    log_level=LogLevel.INFO   # Control output verbosity
)

# Access data directly (no .select().get() needed!)
data = multigas.six_hours

# Chain filters elegantly
filtered = (data
    .filter_date_range('2024-05-17', '2024-06-18')
    .filter_column('Status_Flag', '==', 0)
    .filter_columns_between('Avg_CO2_lowpass', 250, 500)
    .select_columns(['Avg_CO2_lowpass', 'Avg_SO2', 'Avg_H2S']))

# Access as pandas DataFrame
df = filtered.df  # Property, not method!

# Analyze
print(f"CO2 mean: {df['Avg_CO2_lowpass'].mean():.2f} ppm")
print(f"Records: {len(df)}")

# Save results
filtered.save("output/filtered_data.csv")
```

---

## 📊 Performance

Tested with 1 year of real volcanic monitoring data from Tangkuban Parahu:

| Metric | v1.x | v2.0 | Improvement |
|--------|------|------|-------------|
| **First load** | ~6s | 5.92s | Baseline |
| **Cached load** | ~6s | 0.04s | **136x faster** ✨ |
| **Memory usage** | ~187 MB | 62 MB | **67% reduction** 💾 |
| **Filtering** | Copies data | Views only | **Near instant** ⚡ |

---

## 📖 Features & Examples

### 1. Loading Data

```python
from magma_multigas import MultiGas, LogLevel

# Load multiple datasets at once
multigas = MultiGas(
    two_seconds='path/to/two_seconds.dat',
    six_hours='path/to/six_hours.dat',
    one_minute='path/to/one_minute.dat',
    zero='path/to/zero.dat',
    span='path/to/span.dat',  # Optional
    normalize=True,
    cache_normalized=True,
    log_level=LogLevel.WARN  # Only show warnings
)

# View summary
print(multigas.summary())
```

**Output:**
```
  dataset_type    rows  columns          start_date            end_date station
0    six_hours    1179       39 2024-03-09 00:30:00 2025-03-03 00:30:00    TANG
1   one_minute  424322       17 2024-03-08 23:51:00 2025-03-03 03:45:00    TANG
2         zero    1178       31 2024-03-09 05:47:59 2025-03-02 23:47:59    TANG
```

### 2. Accessing Datasets

```python
# Direct property access (v2.0)
six_hours = multigas.six_hours
one_minute = multigas.one_minute
zero = multigas.zero

# Check dataset info
print(f"Shape: {six_hours.shape}")
print(f"Date range: {six_hours.date_range}")
print(f"Columns: {six_hours.columns}")
print(f"Station: {six_hours.metadata.station}")
```

### 3. Filtering Data

#### Date Range Filter
```python
# Filter by date range
filtered = data.filter_date_range('2024-05-01', '2024-06-30')
print(f"Filtered: {len(data)} → {len(filtered)} rows")
```

#### Column Value Filter
```python
# Filter by column value
normal_data = filtered.filter_column('Status_Flag', '==', 0)
print(f"Normal operation: {len(normal_data)} rows")
```

#### Range Filter
```python
# Filter values between min and max
co2_range = normal_data.filter_columns_between('Avg_CO2_lowpass', 250, 500)
print(f"CO2 in range: {len(co2_range)} rows")
```

#### Column Selection
```python
# Select specific columns
selected = co2_range.select_columns([
    'Avg_CO2_lowpass',
    'Avg_SO2',
    'Avg_H2S',
    'Avg_H2O'
])
print(f"Selected columns: {selected.columns}")
```

### 4. Method Chaining

```python
# Combine all filters in one elegant chain
result = (multigas.six_hours
    .filter_date_range('2024-05-01', '2024-06-30')
    .filter_column('Status_Flag', '==', 0)
    .filter_columns_between('Avg_CO2_lowpass', 250, 500)
    .select_columns(['Avg_CO2_lowpass', 'Avg_SO2', 'Avg_H2S']))

# Original data unchanged (immutability!)
print(f"Original: {len(multigas.six_hours)} rows")
print(f"Filtered: {len(result)} rows")
```

### 5. Working with DataFrames

```python
# Access as pandas DataFrame
df = result.df  # Property, not method

# Standard pandas operations
print(df.head())
print(df.describe())
print(df.info())

# Analyze
co2_mean = df['Avg_CO2_lowpass'].mean()
co2_std = df['Avg_CO2_lowpass'].std()
print(f"CO2: {co2_mean:.2f} ± {co2_std:.2f} ppm")
```

### 6. Saving Results

```python
# Save to CSV (infers format from extension)
result.save("output/filtered_data.csv")

# Save to Excel
result.save("output/filtered_data.xlsx")

# Save to Parquet (efficient for large datasets)
result.save("output/filtered_data.parquet")

# Specify format explicitly
result.save("output/data.txt", file_format=FileFormat.CSV)
```

### 7. Wind Direction Analysis

```python
# Add wind direction column based on degrees
with_wind = data.add_wind_direction(
    direction_col='Avg_Wind_Direction',
    direction_count=8,  # 8 or 16 directions
    return_as_code=True  # Return 'N', 'NE', etc.
)

# Filter by wind direction
north_wind = with_wind.filter_column('wind_direction', '==', 'N')
print(f"North wind data: {len(north_wind)} rows")
```

### 8. Cache Management

```python
# Clear cache if data files are updated
count = multigas.clear_cache()
print(f"Cleared {count} cache file(s)")

# Disable caching if needed
multigas = MultiGas(
    six_hours=path,
    cache_normalized=False  # No caching
)
```

---

## 🎓 Advanced Usage

### Working with Multiple Datasets

```python
# Filter all datasets by the same date range
filtered_all = multigas.filter_all('2024-05-01', '2024-06-30')

# Access as collection
collection = multigas.as_collection()
for dataset_type, dataset in collection.items():
    print(f"{dataset_type}: {len(dataset)} rows")

# Extract to dictionary of DataFrames
dataframes = multigas.extract_daily('2024-05-01', '2024-06-30')
print(dataframes.keys())  # dict_keys(['six_hours', 'one_minute', 'zero'])
```

### Custom Analysis Pipeline

```python
def analyze_gas_ratios(data, start_date, end_date):
    """Analyze CO2/SO2 ratios for a date range."""

    # Filter and select relevant data
    filtered = (data
        .filter_date_range(start_date, end_date)
        .filter_column('Status_Flag', '==', 0)
        .select_columns([
            'Avg_CO2_lowpass',
            'Avg_SO2',
            'Avg_H2S',
            'Avg_CO2_SO2_ratio'
        ]))

    # Get DataFrame for analysis
    df = filtered.df

    # Calculate statistics
    stats = {
        'co2_mean': df['Avg_CO2_lowpass'].mean(),
        'co2_std': df['Avg_CO2_lowpass'].std(),
        'so2_mean': df['Avg_SO2'].mean(),
        'ratio_mean': df['Avg_CO2_SO2_ratio'].mean(),
        'n_samples': len(df)
    }

    return stats, filtered

# Use the pipeline
stats, filtered_data = analyze_gas_ratios(
    multigas.six_hours,
    '2024-05-01',
    '2024-06-30'
)
print(stats)
```

---

## 📚 Data Types

The package processes 5 types of data files from Campbell Scientific loggers:

| Type | Frequency | Records/Day | Description |
|------|-----------|-------------|-------------|
| `two_seconds` | 2 seconds | 43,200 | High-frequency gas measurements |
| `six_hours` | 6 hours | 4 | Averaged gas ratios and statistics |
| `one_minute` | 1 minute | 1,440 | Meteorological data |
| `zero` | Variable | ~4 | Calibration zero measurements |
| `span` | Variable | ~4 | Calibration span measurements (optional) |

---

## 🔄 Migration from v1.x

### API Changes

| v1.x | v2.0 | Change |
|------|------|--------|
| `mg.select('six_hours').get()` | `mg.six_hours` | Direct property access |
| `data.get()` | `data.df` | Property instead of method |
| `where_date_between(...)` | `filter_date_range(...)` | Renamed for clarity |
| `where('col', '==', val)` | `filter_column('col', '==', val)` | Renamed for clarity |
| `where_values_between(...)` | `filter_columns_between(...)` | Renamed for clarity |
| `save_as(file_type='csv')` | `save("output.csv")` | Simplified |
| `overwrite=True` | `normalize=True, cache_normalized=True` | Explicit parameters |

### Example Migration

**Before (v1.x):**
```python
from magma_multigas import MultiGas

mg = MultiGas(six_hours=path, overwrite=True)
data = mg.select('six_hours').get()
filtered = (data
    .where_date_between('2024-05-01', '2024-06-30')
    .where('Status_Flag', '==', 0)
    .get())
df = filtered.get()
filtered.save_as(file_type='csv')
```

**After (v2.0):**
```python
from magma_multigas import MultiGas, LogLevel

mg = MultiGas(six_hours=path, log_level=LogLevel.INFO)
data = mg.six_hours
filtered = (data
    .filter_date_range('2024-05-01', '2024-06-30')
    .filter_column('Status_Flag', '==', 0))
df = filtered.df
filtered.save("output.csv")
```

**Benefits:** Cleaner, faster, less memory, type-safe!

---

## 🐛 Troubleshooting

### Import Error
```python
# ✓ Correct
from magma_multigas import MultiGas, LogLevel

# ✗ Old v1.x (still works but deprecated)
from magma_multigas import MultiGasData, Query
```

### Cache Issues
```python
# Clear cache if data not updating
mg.clear_cache()

# Or disable caching
mg = MultiGas(path, cache_normalized=False)
```

### Too Much Logging
```python
# Reduce verbosity
mg = MultiGas(path, log_level=LogLevel.ERROR)
```

### Memory Issues
```python
# Filter early to reduce size
filtered = (data
    .filter_date_range('2024-05-01', '2024-05-31')  # Reduce first
    .select_columns(['CO2', 'SO2', 'H2S'])  # Then select
)
```

---

## 📝 TOA5 Format Support

The package automatically handles Campbell Scientific TOA5 format:

```
"TOA5","TANG_RTU","CR1000X","52172","CR1000X.Std.07.01","CPU:..."
"TIMESTAMP","RECORD","Site_Name","Duty_Cycle",...
"TS","RN","","","",...
"","","Smp","Smp","Smp",...
"2024-03-09 00:30:00",0,"MGSXXX_YYYY",6,0,...
```

Features:
- ✅ Automatic header detection
- ✅ Metadata extraction (station, logger, firmware, program)
- ✅ Unit and sampling type parsing
- ✅ TIMESTAMP index conversion

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/martanto/magma-multigas.git
cd magma-multigas

# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Format code
uv run ruff format src/

# Lint code
uv run ruff check src/ --fix
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Martanto**
- Email: martanto@live.com
- GitHub: [@martanto](https://github.com/martanto)

---

## 🙏 Acknowledgments

- **CVGHM** (Center for Volcanology and Geological Hazard Mitigation) - Indonesia
- **USGS** (United States Geological Survey)
- Campbell Scientific for datalogger systems

---

## 📚 Documentation

For detailed documentation, see:
- [CLAUDE.md](CLAUDE.md) - Developer guide and architecture
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) - Implementation progress
- [REAL_DATA_TEST_RESULTS.md](REAL_DATA_TEST_RESULTS.md) - Performance benchmarks

---

## 🔗 Links

- **GitHub:** https://github.com/martanto/magma-multigas
- **PyPI:** https://pypi.org/project/magma-multigas/
- **Issues:** https://github.com/martanto/magma-multigas/issues

---

## ⭐ Show Your Support

If you find this package useful, please consider giving it a star on GitHub!

---

**Note:** v2.0 is a major redesign focused on performance, usability, and code quality. For v1.x documentation, see the [v1.x branch](https://github.com/martanto/magma-multigas/tree/v1.x).
