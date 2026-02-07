#!/usr/bin/env python
"""Test plotting module with real Tangkuban Parahu data."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from magma_multigas import (
    AvailabilityPlotter,
    MultiGas,
    PlotConfig,
    TimeSeriesPlotter,
)

# Data paths
DATA_DIR = Path("D:/Data/Multigas/Tangkuban Parahu")
SIX_HOURS = DATA_DIR / "TANG_RTU_Data_6Hr.dat"
TWO_SECONDS = DATA_DIR / "TANG_RTU_ChemData_Sec2.dat"
ONE_MINUTE = DATA_DIR / "TANG_RTU_Met_1Min.dat"

# Output directory
FIGURES_DIR = Path("figures/test_v2")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def test_availability_plots():
    """Test AvailabilityPlotter with six_hours data."""
    print("\n" + "=" * 60)
    print("Testing AvailabilityPlotter")
    print("=" * 60)

    # Load data
    print("\n1. Loading six_hours data...")
    mg = MultiGas(six_hours=str(SIX_HOURS))
    data = mg.six_hours

    print(f"   [OK] Loaded {len(data)} records")
    print(f"   Date range: {data.date_range[0]} to {data.date_range[1]}")

    # Create plotter
    print("\n2. Creating AvailabilityPlotter...")
    plotter = AvailabilityPlotter(data)
    print("   [OK] Plotter created")

    # Test 1: Calendar heatmap
    print("\n3. Testing calendar heatmap...")
    try:
        plotter.plot_calendar_heatmap()
        output_path = plotter.save(str(FIGURES_DIR / "availability_calendar.png"))
        print(f"   [OK] Saved to: {output_path}")
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False

    # Test 2: Daily counts
    print("\n4. Testing daily counts plot...")
    try:
        plotter = AvailabilityPlotter(data)  # New instance
        plotter.plot_daily_counts()
        output_path = plotter.save(str(FIGURES_DIR / "availability_daily.png"))
        print(f"   [OK] Saved to: {output_path}")
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False

    # Test 3: Completeness bar
    print("\n5. Testing completeness bar chart...")
    try:
        plotter = AvailabilityPlotter(data)  # New instance
        plotter.plot_completeness_bar(threshold=0.5)
        output_path = plotter.save(str(FIGURES_DIR / "availability_completeness.png"))
        print(f"   [OK] Saved to: {output_path}")
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False

    # Test 4: Missing patterns
    print("\n6. Testing missing patterns heatmap...")
    try:
        plotter = AvailabilityPlotter(data)  # New instance
        plotter.plot_missing_patterns(max_columns=15)
        output_path = plotter.save(str(FIGURES_DIR / "availability_missing.png"))
        print(f"   [OK] Saved to: {output_path}")
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False

    # Test 5: Get statistics
    print("\n7. Testing get_statistics()...")
    try:
        stats = plotter.get_statistics()
        print(f"   [OK] Retrieved stats for {len(stats)} columns")
        print("\n   Top 5 most complete columns:")
        for _, row in stats.head(5).iterrows():
            print(
                f"      - {row['column']}: {row['completeness_pct']:.1f}% "
                f"({row['available']}/{row['total_records']})"
            )
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False

    return True


def test_timeseries_plots():
    """Test TimeSeriesPlotter with six_hours data."""
    print("\n" + "=" * 60)
    print("Testing TimeSeriesPlotter")
    print("=" * 60)

    # Load data
    print("\n1. Loading six_hours data...")
    mg = MultiGas(six_hours=str(SIX_HOURS))
    data = mg.six_hours

    print(f"   [OK] Loaded {len(data)} records")

    # Filter to specific date range
    print("\n2. Filtering to May-July 2024...")
    filtered = data.filter_date_range("2024-05-17", "2024-07-24")
    print(f"   [OK] Filtered to {len(filtered)} records")

    # Create plotter
    print("\n3. Creating TimeSeriesPlotter...")
    plotter = TimeSeriesPlotter(filtered)
    print("   [OK] Plotter created")

    # Test 1: CO2/SO2/H2S dual-axis plot
    print("\n4. Testing CO2/SO2/H2S dual-axis plot...")
    try:
        plotter.plot_co2_so2_h2s(
            co2_col="Avg_CO2_lowpass",
            so2_col="Avg_SO2",
            h2s_col="Avg_H2S",
            plot_as_individual=False,
        )
        output_path = plotter.save(str(FIGURES_DIR / "timeseries_gas_dual.png"))
        print(f"   [OK] Saved to: {output_path}")
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False

    # Test 2: CO2/SO2/H2S individual plots
    print("\n5. Testing CO2/SO2/H2S individual plots...")
    try:
        plotter = TimeSeriesPlotter(filtered)  # New instance
        plotter.plot_co2_so2_h2s(
            co2_col="Avg_CO2_lowpass",
            so2_col="Avg_SO2",
            h2s_col="Avg_H2S",
            plot_as_individual=True,
        )
        output_path = plotter.save(str(FIGURES_DIR / "timeseries_gas_individual.png"))
        print(f"   [OK] Saved to: {output_path}")
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False

    # Test 3: Gas ratios
    print("\n6. Testing gas ratios plot...")
    try:
        plotter = TimeSeriesPlotter(filtered)  # New instance
        plotter.plot_gas_ratios(
            ratios=[
                "Avg_CO2_H2S_ratio",
                "Avg_H2O_CO2_ratio",
                "Avg_H2S_SO2_ratio",
                "Avg_CO2_S_tot_ratio",
            ],
            plot_regression=True,
        )
        output_path = plotter.save(str(FIGURES_DIR / "timeseries_ratios.png"))
        print(f"   [OK] Saved to: {output_path}")
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False

    # Test 4: Custom columns plot (separate)
    print("\n7. Testing custom columns plot (separate)...")
    try:
        plotter = TimeSeriesPlotter(filtered)  # New instance
        plotter.plot_columns(
            columns=["Avg_CO2_lowpass", "Avg_H2O"], separate=True, title="CO2 and H2O"
        )
        output_path = plotter.save(str(FIGURES_DIR / "timeseries_custom_separate.png"))
        print(f"   [OK] Saved to: {output_path}")
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False

    # Test 5: Custom columns plot (combined)
    print("\n8. Testing custom columns plot (combined)...")
    try:
        plotter = TimeSeriesPlotter(filtered)  # New instance
        plotter.plot_columns(
            columns=["Avg_CO2_lowpass", "Avg_H2O"],
            separate=False,
            title="CO2 and H2O Combined",
        )
        output_path = plotter.save(str(FIGURES_DIR / "timeseries_custom_combined.png"))
        print(f"   [OK] Saved to: {output_path}")
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False

    return True


def test_custom_config():
    """Test custom PlotConfig."""
    print("\n" + "=" * 60)
    print("Testing Custom PlotConfig")
    print("=" * 60)

    # Load data
    print("\n1. Loading six_hours data...")
    mg = MultiGas(six_hours=str(SIX_HOURS))
    data = mg.six_hours.filter_date_range("2024-05-17", "2024-07-24")
    print(f"   [OK] Loaded and filtered to {len(data)} records")

    # Create custom config
    print("\n2. Creating custom PlotConfig...")
    config = PlotConfig(
        width=14,
        height=6,
        dpi=150,
        style="darkgrid",
        context="talk",
        font_scale=1.2,
    )
    print("   [OK] Custom config created")

    # Test with custom config
    print("\n3. Creating plot with custom config...")
    try:
        plotter = TimeSeriesPlotter(data, config=config)
        plotter.plot_co2_so2_h2s()
        output_path = plotter.save(str(FIGURES_DIR / "timeseries_custom_config.png"))
        print(f"   [OK] Saved to: {output_path}")
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False

    return True


def test_method_chaining():
    """Test method chaining pattern."""
    print("\n" + "=" * 60)
    print("Testing Method Chaining")
    print("=" * 60)

    # Load data
    print("\n1. Testing method chaining pattern...")
    mg = MultiGas(six_hours=str(SIX_HOURS))

    try:
        # Chain: load → filter → plot → save
        output_path = (
            TimeSeriesPlotter(
                mg.six_hours.filter_date_range("2024-05-17", "2024-06-30")
            )
            .plot_co2_so2_h2s(plot_as_individual=False)
            .save(str(FIGURES_DIR / "timeseries_chained.png"))
        )
        print(f"   [OK] Chained operations successful")
        print(f"   [OK] Saved to: {output_path}")
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("magma-multigas v2.0 - Plotting Module Test")
    print("=" * 60)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Output directory: {FIGURES_DIR}")

    results = []

    # Run tests
    results.append(("AvailabilityPlotter", test_availability_plots()))
    results.append(("TimeSeriesPlotter", test_timeseries_plots()))
    results.append(("Custom PlotConfig", test_custom_config()))
    results.append(("Method Chaining", test_method_chaining()))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed}/{total} test suites passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[FAILED] {total - passed} test suite(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
