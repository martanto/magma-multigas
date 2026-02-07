# Plotting Quick Start Guide - v2.0

A quick reference for using the new plotting module in magma-multigas v2.0.

---

## Installation & Imports

```python
from magma_multigas import (
    MultiGas,
    TimeSeriesPlotter,
    AvailabilityPlotter,
    PlotConfig
)
```

---

## Time Series Plotting

### 1. Basic CO2/SO2/H2S Plot (Dual-Axis)

```python
# Load data
mg = MultiGas(six_hours="data/TANG_RTU_Data_6Hr.dat")
data = mg.six_hours.filter_date_range("2024-05-17", "2024-07-24")

# Plot
plotter = TimeSeriesPlotter(data)
plotter.plot_co2_so2_h2s()
plotter.save("figures/gas_plot.png")
```

### 2. Individual Gas Plots (Stacked)

```python
plotter = TimeSeriesPlotter(data)
plotter.plot_co2_so2_h2s(plot_as_individual=True)
plotter.save("figures/gas_stacked.png")
```

### 3. Custom Column Names

```python
# Works with ANY column names!
plotter = TimeSeriesPlotter(data)
plotter.plot_co2_so2_h2s(
    co2_col="My_CO2_Column",
    so2_col="My_SO2_Column",
    h2s_col="My_H2S_Column"
)
plotter.save("figures/custom_names.png")
```

### 4. Gas Ratios with Trend Lines

```python
plotter = TimeSeriesPlotter(data)
plotter.plot_gas_ratios(
    ratios=[
        "Avg_CO2_H2S_ratio",
        "Avg_H2O_CO2_ratio",
        "Avg_H2S_SO2_ratio",
        "Avg_CO2_S_tot_ratio"
    ],
    plot_regression=True
)
plotter.save("figures/ratios.png")
```

### 5. Custom Columns (Any Variables)

```python
# Separate plots
plotter = TimeSeriesPlotter(data)
plotter.plot_columns(
    columns=["Avg_CO2_lowpass", "Avg_H2O", "Avg_SO2"],
    separate=True
)
plotter.save("figures/custom_separate.png")

# Combined plot
plotter = TimeSeriesPlotter(data)
plotter.plot_columns(
    columns=["Avg_CO2_lowpass", "Avg_H2O"],
    separate=False
)
plotter.save("figures/custom_combined.png")
```

### 6. Method Chaining

```python
# Load → Filter → Plot → Save in one chain
output = (
    TimeSeriesPlotter(
        MultiGas(six_hours="data.dat")
        .six_hours
        .filter_date_range("2024-05-17", "2024-07-24")
    )
    .plot_co2_so2_h2s()
    .save("figures/gas_plot.png")
)
```

---

## Data Availability Plotting

### 1. Calendar Heatmap

```python
mg = MultiGas(six_hours="data.dat")
plotter = AvailabilityPlotter(mg.six_hours)

plotter.plot_calendar_heatmap()
plotter.save("figures/calendar.png")
```

### 2. Daily Record Counts

```python
plotter = AvailabilityPlotter(mg.six_hours)
plotter.plot_daily_counts()
plotter.save("figures/daily_counts.png")
```

### 3. Completeness Bar Chart

```python
plotter = AvailabilityPlotter(mg.six_hours)
plotter.plot_completeness_bar(threshold=0.5)  # Only show >50% complete
plotter.save("figures/completeness.png")
```

### 4. Missing Data Patterns

```python
plotter = AvailabilityPlotter(mg.six_hours)
plotter.plot_missing_patterns(max_columns=15)
plotter.save("figures/missing.png")
```

### 5. Get Completeness Statistics

```python
plotter = AvailabilityPlotter(mg.six_hours)
stats = plotter.get_statistics()

print(stats.head())
#    column  total_records  available  missing  completeness_pct
# 0  RECORD           1179       1179        0             100.0
# 1  Status           1179       1179        0             100.0
```

---

## Custom Plot Configuration

### 1. Basic Custom Config

```python
# Create custom configuration
config = PlotConfig(
    width=14,
    height=6,
    dpi=150,
    style="darkgrid",
    context="talk",
    font_scale=1.2
)

# Use in plotter
plotter = TimeSeriesPlotter(data, config=config)
plotter.plot_co2_so2_h2s()
plotter.save("figures/custom_style.png")
```

### 2. Available Styles

```python
# Seaborn styles
styles = ["whitegrid", "darkgrid", "white", "dark", "ticks"]

# Seaborn contexts
contexts = ["paper", "notebook", "talk", "poster"]

# Example
config = PlotConfig(style="ticks", context="poster")
```

### 3. Font Sizes

```python
config = PlotConfig(
    title_fontsize=16,
    label_fontsize=14,
    tick_fontsize=12,
    legend_fontsize=11
)
```

### 4. Figure Dimensions

```python
# For presentations
config = PlotConfig(width=16, height=9, dpi=150)

# For publications
config = PlotConfig(width=12, height=4, dpi=300)

# For posters
config = PlotConfig(width=18, height=12, dpi=300)
```

### 5. Copy and Modify Config

```python
# Start with defaults
config = PlotConfig()

# Create modified copy
custom_config = config.copy(
    width=14,
    dpi=150,
    style="darkgrid"
)
```

---

## Advanced Examples

### 1. Multiple Plots with Same Data

```python
mg = MultiGas(six_hours="data.dat")
data = mg.six_hours.filter_date_range("2024-05-17", "2024-07-24")

# Create multiple plots
plots = [
    ("gas_dual.png", lambda p: p.plot_co2_so2_h2s(plot_as_individual=False)),
    ("gas_stacked.png", lambda p: p.plot_co2_so2_h2s(plot_as_individual=True)),
    ("ratios.png", lambda p: p.plot_gas_ratios()),
]

for filename, plot_func in plots:
    plotter = TimeSeriesPlotter(data)
    plot_func(plotter)
    plotter.save(f"figures/{filename}")
```

### 2. Custom Y-Axis Ranges

```python
plotter = TimeSeriesPlotter(data)
plotter.plot_co2_so2_h2s(
    y_left_min=0,
    y_left_max=500,    # CO2 range
    y_right_min=-1,
    y_right_max=10     # SO2/H2S range
)
plotter.save("figures/custom_ranges.png")
```

### 3. Availability Analysis Workflow

```python
mg = MultiGas(six_hours="data.dat")
data = mg.six_hours

# Create plotter
avail = AvailabilityPlotter(data)

# Generate all plots
plots = {
    "calendar": lambda: avail.plot_calendar_heatmap(),
    "daily": lambda: avail.plot_daily_counts(),
    "completeness": lambda: avail.plot_completeness_bar(threshold=0.5),
    "missing": lambda: avail.plot_missing_patterns()
}

for name, plot_func in plots.items():
    plotter = AvailabilityPlotter(data)
    plot_func()
    plotter.save(f"figures/availability_{name}.png")

# Get statistics
stats = avail.get_statistics()
stats.to_csv("output/completeness_stats.csv", index=False)
```

### 4. Batch Processing Multiple Stations

```python
stations = {
    "TANG": "data/TANG_RTU_Data_6Hr.dat",
    "PAKA": "data/PAKA_RTU_Data_6Hr.dat",
}

for station_name, data_file in stations.items():
    mg = MultiGas(six_hours=data_file)
    data = mg.six_hours

    # Time series
    plotter = TimeSeriesPlotter(data)
    plotter.plot_co2_so2_h2s()
    plotter.save(f"figures/{station_name}_gas.png")

    # Availability
    avail = AvailabilityPlotter(data)
    avail.plot_completeness_bar()
    avail.save(f"figures/{station_name}_completeness.png")
```

---

## Tips & Best Practices

### 1. Always Create New Plotter for Each Plot

```python
# ✅ CORRECT - New plotter for each plot
plotter1 = TimeSeriesPlotter(data)
plotter1.plot_co2_so2_h2s()
plotter1.save("plot1.png")

plotter2 = TimeSeriesPlotter(data)
plotter2.plot_gas_ratios()
plotter2.save("plot2.png")

# ❌ WRONG - Reusing plotter
plotter = TimeSeriesPlotter(data)
plotter.plot_co2_so2_h2s()
plotter.save("plot1.png")
plotter.plot_gas_ratios()  # ← Will fail!
```

### 2. Use Method Chaining for Concise Code

```python
# ✅ CORRECT - Fluent chaining
(TimeSeriesPlotter(data)
    .plot_co2_so2_h2s()
    .save("output.png"))

# ❌ VERBOSE - Intermediate variables
plotter = TimeSeriesPlotter(data)
plotter.plot_co2_so2_h2s()
output = plotter.save("output.png")
```

### 3. Filter Before Plotting

```python
# ✅ CORRECT - Filter first, then plot
data = mg.six_hours.filter_date_range("2024-05-17", "2024-07-24")
plotter = TimeSeriesPlotter(data)

# ❌ WRONG - Don't filter inside plotting
plotter = TimeSeriesPlotter(mg.six_hours)
# No way to filter after TimeSeriesPlotter creation
```

### 4. Use Custom Config for Consistency

```python
# ✅ CORRECT - Reuse config for consistent styling
config = PlotConfig(dpi=150, style="darkgrid")

plotter1 = TimeSeriesPlotter(data1, config=config)
plotter1.plot_co2_so2_h2s().save("plot1.png")

plotter2 = TimeSeriesPlotter(data2, config=config)
plotter2.plot_gas_ratios().save("plot2.png")
```

### 5. Specify Column Names for Non-Standard Data

```python
# If your columns have different names
plotter = TimeSeriesPlotter(data)
plotter.plot_co2_so2_h2s(
    co2_col="carbon_dioxide",  # Your column name
    so2_col="sulfur_dioxide",
    h2s_col="hydrogen_sulfide"
)
```

---

## Common Patterns

### Pattern 1: Quick Visualization

```python
# One-liner for quick plots
TimeSeriesPlotter(
    MultiGas(six_hours="data.dat").six_hours
).plot_co2_so2_h2s().save("quick.png")
```

### Pattern 2: Publication-Quality Plots

```python
# High-quality config
pub_config = PlotConfig(
    width=12,
    height=4,
    dpi=300,
    style="ticks",
    context="paper"
)

# Create publication plot
(TimeSeriesPlotter(data, config=pub_config)
    .plot_co2_so2_h2s(plot_as_individual=True)
    .save("manuscript_figure1.png"))
```

### Pattern 3: Data Quality Report

```python
# Generate complete availability report
mg = MultiGas(six_hours="data.dat")
data = mg.six_hours

reports = [
    ("calendar", "plot_calendar_heatmap", {}),
    ("daily", "plot_daily_counts", {}),
    ("completeness", "plot_completeness_bar", {"threshold": 0.5}),
    ("missing", "plot_missing_patterns", {"max_columns": 15})
]

for name, method, kwargs in reports:
    plotter = AvailabilityPlotter(data)
    getattr(plotter, method)(**kwargs)
    plotter.save(f"reports/{name}.png")

# Save statistics
stats = AvailabilityPlotter(data).get_statistics()
stats.to_csv("reports/completeness.csv", index=False)
```

---

## Error Handling

### 1. Missing Columns

```python
# The plotter will raise ValueError if columns don't exist
try:
    plotter = TimeSeriesPlotter(data)
    plotter.plot_columns(["NonExistent_Column"])
except ValueError as e:
    print(f"Column error: {e}")
```

### 2. Empty Dataset

```python
# Check data before plotting
if len(data) > 0:
    plotter = TimeSeriesPlotter(data)
    plotter.plot_co2_so2_h2s()
else:
    print("No data to plot")
```

### 3. Missing Data in Columns

```python
# Some columns might have no data
# Availability plotter handles this automatically
plotter = AvailabilityPlotter(data)
try:
    plotter.plot_missing_patterns()
    plotter.save("missing.png")
except ValueError as e:
    print(f"No missing data found: {e}")
```

---

## Next Steps

- See **PHASE3_PLOTTING_SUMMARY.md** for complete technical documentation
- See **CLAUDE.md** for development guidelines
- See **README.md** for full package documentation
- See **test_v2_plotting.py** for comprehensive examples

---

*Generated for magma-multigas v2.0 | February 7, 2026*
