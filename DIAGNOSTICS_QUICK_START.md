# Diagnostics Quick Start Guide - v2.0

A quick reference for using the diagnostics module in magma-multigas v2.0.

---

## Installation & Imports

```python
from magma_multigas import (
    MultiGas,
    DataDiagnostics
)
```

---

## Quick Start

### 1. Basic Setup

```python
# Load data
mg = MultiGas(six_hours="data/TANG_RTU_Data_6Hr.dat")

# Create diagnostics
diag = DataDiagnostics(mg.six_hours)
```

---

## Completeness Analysis

### Check Overall Completeness

```python
# Analyze completeness
report = diag.analyze_completeness(expected_freq="6h")

print(f"Completeness: {report.completeness_pct:.1f}%")
print(f"Records: {report.total_records}/{report.expected_records}")
print(f"Date range: {report.date_range[0]} to {report.date_range[1]}")
```

### View Column Statistics

```python
# Get per-column stats
print(report.column_stats.head(10))

# Save to CSV
report.column_stats.to_csv("completeness_stats.csv", index=False)
```

### Check Missing Periods

```python
# Show missing data periods
print(f"Missing periods: {len(report.missing_periods)}")

for start, end in report.missing_periods[:5]:
    print(f"  {start} to {end}")
```

---

## Gap Detection

### Detect All Gaps

```python
# Find gaps with auto threshold (2x median interval)
report = diag.detect_gaps()

print(f"Total gaps: {report.total_gaps}")
print(f"Longest gap: {report.longest_gap}")
```

### Filter by Minimum Duration

```python
from datetime import timedelta

# Only show gaps longer than 12 hours
report = diag.detect_gaps(min_gap=timedelta(hours=12))

print(f"Gaps > 12 hours: {report.total_gaps}")
```

### View Gap Details

```python
# Show all gaps with details
for start, end, duration in report.gaps:
    print(f"Gap: {start} to {end} (duration: {duration})")

# Gap statistics
print(f"Mean gap: {report.gap_summary['mean']}")
print(f"Median gap: {report.gap_summary['median']}")
```

---

## Calibration Analysis

### Check Calibration Schedule

```python
# Analyze calibration periods
report = diag.analyze_calibrations()

print(f"Zero calibrations: {report.zero_count}")
print(f"Span calibrations: {report.span_count}")

if report.calibration_frequency:
    print(f"Calibration every {report.calibration_frequency.days} days")
```

### View Calibration Periods

```python
# Show zero calibration periods
print("\nZero calibration periods:")
for start, end in report.zero_periods:
    duration = end - start
    print(f"  {start} to {end} ({duration})")

# Show span calibration periods
print("\nSpan calibration periods:")
for start, end in report.span_periods:
    duration = end - start
    print(f"  {start} to {end} ({duration})")
```

---

## Anomaly Detection

### Z-Score Method (Default)

```python
# Detect outliers using z-score (3 standard deviations)
report = diag.detect_anomalies(
    column="Avg_CO2_lowpass",
    method="zscore",
    threshold=3.0
)

print(f"Anomalies found: {report.anomaly_count}")
print(f"Percentage: {report.anomaly_count/len(diag.df)*100:.1f}%")
```

### IQR Method (Robust)

```python
# Use IQR method for skewed data
report = diag.detect_anomalies(
    column="Avg_CO2_lowpass",
    method="iqr",
    threshold=1.5  # Standard IQR multiplier
)

print(f"Anomalies (IQR): {report.anomaly_count}")
```

### MAD Method (Very Robust)

```python
# Use MAD for data with many outliers
report = diag.detect_anomalies(
    column="Avg_CO2_lowpass",
    method="mad",
    threshold=3.0
)

print(f"Anomalies (MAD): {report.anomaly_count}")
```

### View Anomaly Details

```python
# Get anomaly statistics
print(f"Mean value: {report.summary_stats['mean']:.2f}")
print(f"Std dev: {report.summary_stats['std']:.2f}")
print(f"Min value: {report.summary_stats['min']:.2f}")
print(f"Max value: {report.summary_stats['max']:.2f}")

# Get anomaly indices (for filtering)
anomaly_df = diag.df.loc[report.anomaly_indices]
print(anomaly_df.head())
```

---

## Statistical Summary

### Summary for All Numeric Columns

```python
# Get stats for all numeric columns
summary = diag.get_statistical_summary()

print(summary)
```

### Summary for Specific Columns

```python
# Analyze specific gas columns
gas_cols = ["Avg_CO2_lowpass", "Avg_SO2", "Avg_H2S", "Avg_H2O"]
summary = diag.get_statistical_summary(columns=gas_cols)

print(summary[['column', 'mean', 'std', 'min', 'max']])
```

### Save Summary

```python
# Save to CSV
summary.to_csv("statistical_summary.csv", index=False)

# Save to Excel
summary.to_excel("statistical_summary.xlsx", index=False)
```

---

## Health Report

### Generate Complete Health Report

```python
# Get overall health assessment
health = diag.generate_health_report()

print(f"\nHealth Score: {health['health_score']:.1f}/100\n")

# Completeness
print(f"Completeness: {health['completeness']['pct']:.1f}%")
print(f"  Records: {health['completeness']['records']}/{health['completeness']['expected']}")

# Gaps
print(f"\nGaps: {health['gaps']['count']}")
print(f"  Longest: {health['gaps']['longest_hours']:.1f} hours")

# Calibrations
print(f"\nCalibrations:")
print(f"  Zero: {health['calibrations']['zero_count']}")
print(f"  Span: {health['calibrations']['span_count']}")
if health['calibrations']['frequency_days']:
    print(f"  Frequency: {health['calibrations']['frequency_days']} days")

# Recommendations
print(f"\nRecommendations:")
for i, rec in enumerate(health['recommendations'], 1):
    print(f"  {i}. {rec}")
```

### Save Health Report

```python
import json

# Save as JSON
with open("health_report.json", "w") as f:
    json.dump(health, f, indent=2, default=str)

# Or extract specific metrics
completeness_pct = health['completeness']['pct']
health_score = health['health_score']
```

---

## Common Workflows

### Workflow 1: Daily Data Quality Check

```python
from magma_multigas import MultiGas, DataDiagnostics

def daily_quality_check(file_path):
    """Quick daily data quality check."""
    # Load data
    mg = MultiGas(six_hours=file_path)
    diag = DataDiagnostics(mg.six_hours)

    # Generate health report
    health = diag.generate_health_report()

    # Print summary
    print(f"Health Score: {health['health_score']:.1f}/100")

    # Alert if issues found
    if health['health_score'] < 70:
        print("\n⚠️  ISSUES DETECTED:")
        for rec in health['recommendations']:
            print(f"  - {rec}")
    else:
        print("✓ Data quality is good")

    return health

# Run check
health = daily_quality_check("TANG_RTU_Data_6Hr.dat")
```

### Workflow 2: Detailed Analysis for Report

```python
from magma_multigas import MultiGas, DataDiagnostics
import json

def generate_detailed_report(file_path, output_dir):
    """Generate detailed quality report with all analyses."""
    # Load data
    mg = MultiGas(six_hours=file_path)
    diag = DataDiagnostics(mg.six_hours)

    # Run all analyses
    completeness = diag.analyze_completeness(expected_freq="6h")
    gaps = diag.detect_gaps(min_gap=timedelta(hours=12))
    calibrations = diag.analyze_calibrations()
    stats = diag.get_statistical_summary()
    health = diag.generate_health_report()

    # Save results
    completeness.column_stats.to_csv(f"{output_dir}/completeness.csv", index=False)
    stats.to_csv(f"{output_dir}/statistics.csv", index=False)

    with open(f"{output_dir}/health.json", "w") as f:
        json.dump(health, f, indent=2, default=str)

    print(f"Report generated in {output_dir}/")
    print(f"Health Score: {health['health_score']:.1f}/100")

# Generate report
generate_detailed_report("TANG_RTU_Data_6Hr.dat", "reports/")
```

### Workflow 3: Compare Time Periods

```python
from magma_multigas import MultiGas, DataDiagnostics

def compare_periods(file_path, period1, period2):
    """Compare data quality between two time periods."""
    mg = MultiGas(six_hours=file_path)

    # Period 1
    data1 = mg.six_hours.filter_date_range(period1[0], period1[1])
    diag1 = DataDiagnostics(data1)
    health1 = diag1.generate_health_report()

    # Period 2
    data2 = mg.six_hours.filter_date_range(period2[0], period2[1])
    diag2 = DataDiagnostics(data2)
    health2 = diag2.generate_health_report()

    # Compare
    print(f"Period 1 ({period1[0]} to {period1[1]}):")
    print(f"  Health: {health1['health_score']:.1f}/100")
    print(f"  Completeness: {health1['completeness']['pct']:.1f}%")

    print(f"\nPeriod 2 ({period2[0]} to {period2[1]}):")
    print(f"  Health: {health2['health_score']:.1f}/100")
    print(f"  Completeness: {health2['completeness']['pct']:.1f}%")

    # Show improvement/degradation
    change = health2['health_score'] - health1['health_score']
    if change > 0:
        print(f"\n✓ Improvement: +{change:.1f} points")
    else:
        print(f"\n⚠️  Degradation: {change:.1f} points")

# Compare May vs June 2024
compare_periods(
    "TANG_RTU_Data_6Hr.dat",
    ("2024-05-01", "2024-05-31"),
    ("2024-06-01", "2024-06-30")
)
```

### Workflow 4: Anomaly Investigation

```python
from magma_multigas import MultiGas, DataDiagnostics

def investigate_anomalies(file_path, column):
    """Investigate anomalies in a specific column."""
    # Load data (only normal operation)
    mg = MultiGas(six_hours=file_path)
    data = mg.six_hours.filter_column("Status_Flag", "==", 0)

    diag = DataDiagnostics(data)

    # Try different methods
    methods = [
        ("Z-Score (3σ)", "zscore", 3.0),
        ("IQR (1.5)", "iqr", 1.5),
        ("MAD (3.0)", "mad", 3.0),
    ]

    print(f"Anomaly analysis for {column}:\n")

    for name, method, threshold in methods:
        report = diag.detect_anomalies(column, method=method, threshold=threshold)
        pct = report.anomaly_count / len(data) * 100

        print(f"{name}:")
        print(f"  Anomalies: {report.anomaly_count} ({pct:.1f}%)")

        if report.anomaly_count > 0:
            print(f"  Mean: {report.summary_stats['mean']:.2f}")
            print(f"  Max: {report.summary_stats['max']:.2f}")

        print()

# Investigate CO2 anomalies
investigate_anomalies("TANG_RTU_Data_6Hr.dat", "Avg_CO2_lowpass")
```

### Workflow 5: Batch Station Analysis

```python
from pathlib import Path
from magma_multigas import MultiGas, DataDiagnostics
import pandas as pd

def analyze_all_stations(data_dir):
    """Analyze data quality for all stations in directory."""
    results = []

    for file_path in Path(data_dir).glob("*_6Hr.dat"):
        station = file_path.stem.split("_")[0]

        try:
            mg = MultiGas(six_hours=str(file_path))
            diag = DataDiagnostics(mg.six_hours)
            health = diag.generate_health_report()

            results.append({
                "station": station,
                "health_score": health['health_score'],
                "completeness_pct": health['completeness']['pct'],
                "gap_count": health['gaps']['count'],
                "calibrations": health['calibrations']['zero_count']
            })

            print(f"✓ {station}: {health['health_score']:.1f}/100")

        except Exception as e:
            print(f"✗ {station}: {e}")

    # Create summary
    summary = pd.DataFrame(results)
    summary.sort_values("health_score", ascending=False, inplace=True)

    # Save summary
    summary.to_csv("station_summary.csv", index=False)

    return summary

# Analyze all stations
summary = analyze_all_stations("data/")
print("\nStation Summary:")
print(summary)
```

---

## Anomaly Detection Method Comparison

| Method | Best For | Threshold | Sensitivity |
|--------|----------|-----------|-------------|
| **Z-Score** | Normal data | 3.0 (3σ) | High |
| **IQR** | Skewed data | 1.5 (standard) | Medium |
| **MAD** | Data with outliers | 3.0 | Low (robust) |

### When to Use Each:

**Z-Score:**
- Data is approximately normally distributed
- You want to catch subtle anomalies
- Few outliers expected

**IQR:**
- Data is skewed or has heavy tails
- Moderate robustness needed
- Standard statistical approach

**MAD:**
- Data has many outliers
- Maximum robustness needed
- Non-normal distributions

---

## Tips & Best Practices

### 1. Always Specify Expected Frequency

```python
# ✅ GOOD - Explicit frequency
report = diag.analyze_completeness(expected_freq="6h")

# ⚠️  OK - Auto-detect (may fail)
report = diag.analyze_completeness()
```

### 2. Filter Before Analysis

```python
# ✅ GOOD - Analyze only normal operation data
data = mg.six_hours.filter_column("Status_Flag", "==", 0)
diag = DataDiagnostics(data)

# ⚠️  LESS USEFUL - Includes calibration periods
diag = DataDiagnostics(mg.six_hours)
```

### 3. Use Appropriate Anomaly Methods

```python
# For normal data
report = diag.detect_anomalies("col", method="zscore", threshold=3.0)

# For skewed data
report = diag.detect_anomalies("col", method="iqr", threshold=1.5)

# For data with many outliers
report = diag.detect_anomalies("col", method="mad", threshold=3.0)
```

### 4. Save Results for Later Analysis

```python
# Save completeness stats
report.column_stats.to_csv("completeness.csv", index=False)

# Save health report
import json
with open("health.json", "w") as f:
    json.dump(health, f, indent=2, default=str)

# Save statistics
stats.to_csv("statistics.csv", index=False)
```

### 5. Combine with Plotting

```python
from magma_multigas import (
    MultiGas,
    DataDiagnostics,
    AvailabilityPlotter
)

mg = MultiGas(six_hours="data.dat")

# Analyze
diag = DataDiagnostics(mg.six_hours)
completeness = diag.analyze_completeness()

# Visualize
plotter = AvailabilityPlotter(mg.six_hours)
plotter.plot_completeness_bar()
plotter.save("completeness.png")

# Show numbers
print(completeness.column_stats.head())
```

---

## Troubleshooting

### Issue: "Could not infer frequency"

**Solution:** Specify frequency explicitly:
```python
report = diag.analyze_completeness(expected_freq="6h")
```

### Issue: No calibrations detected

**Possible causes:**
1. Status_Flag column not populated
2. Different status flag values used
3. No calibrations performed

**Solution:** Check Status_Flag column:
```python
print(diag.df['Status_Flag'].value_counts())
```

### Issue: Too many anomalies detected

**Solution:** Adjust threshold or use different method:
```python
# Less sensitive
report = diag.detect_anomalies("col", threshold=4.0)  # 4σ instead of 3σ

# More robust method
report = diag.detect_anomalies("col", method="mad")
```

### Issue: Analysis too slow with large datasets

**Solution:** Sample or filter data first:
```python
# Filter to recent period
recent = mg.six_hours.filter_date_range("2024-01-01", "2024-12-31")
diag = DataDiagnostics(recent)
```

---

## Next Steps

- See **PHASE4_ANALYSIS_SUMMARY.md** for complete technical documentation
- See **CLAUDE.md** for development guidelines
- See **README.md** for full package documentation
- See **test_v2_diagnostics.py** for comprehensive examples

---

*Generated for magma-multigas v2.0 | February 8, 2026*
