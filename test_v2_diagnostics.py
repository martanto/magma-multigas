#!/usr/bin/env python
"""Test diagnostics module with real Tangkuban Parahu data."""

import sys
from datetime import timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from magma_multigas import DataDiagnostics, MultiGas

# Data paths
DATA_DIR = Path("D:/Data/Multigas/Tangkuban Parahu")
SIX_HOURS = DATA_DIR / "TANG_RTU_Data_6Hr.dat"
TWO_SECONDS = DATA_DIR / "TANG_RTU_ChemData_Sec2.dat"

# Output directory
OUTPUT_DIR = Path("output/diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_completeness_analysis():
    """Test data completeness analysis."""
    print("\n" + "=" * 60)
    print("Testing Completeness Analysis")
    print("=" * 60)

    # Load data
    print("\n1. Loading six_hours data...")
    mg = MultiGas(six_hours=str(SIX_HOURS))
    data = mg.six_hours
    print(f"   [OK] Loaded {len(data)} records")

    # Create diagnostics
    print("\n2. Creating DataDiagnostics...")
    diag = DataDiagnostics(data)
    print("   [OK] Diagnostics created")

    # Analyze completeness
    print("\n3. Analyzing completeness...")
    try:
        report = diag.analyze_completeness(expected_freq="6h")
        print(f"   [OK] Completeness: {report.completeness_pct:.1f}%")
        print(
            f"   [OK] Records: {report.total_records}/"
            f"{report.expected_records} expected"
        )
        print(f"   [OK] Date range: {report.date_range[0]} to {report.date_range[1]}")
        print(f"   [OK] Missing periods: {len(report.missing_periods)}")

        # Show top columns
        print("\n   Top 5 most complete columns:")
        for _, row in report.column_stats.head(5).iterrows():
            print(f"      - {row['column']}: {row['completeness_pct']:.1f}%")

        # Save column stats
        output_path = OUTPUT_DIR / "completeness_stats.csv"
        report.column_stats.to_csv(output_path, index=False)
        print(f"\n   [OK] Saved column stats to: {output_path}")

        return True
    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback

        traceback.print_exc()
        return False


def test_gap_detection():
    """Test gap detection."""
    print("\n" + "=" * 60)
    print("Testing Gap Detection")
    print("=" * 60)

    # Load data
    print("\n1. Loading six_hours data...")
    mg = MultiGas(six_hours=str(SIX_HOURS))
    data = mg.six_hours
    print(f"   [OK] Loaded {len(data)} records")

    # Create diagnostics
    diag = DataDiagnostics(data)

    # Detect gaps
    print("\n2. Detecting gaps (min 12 hours)...")
    try:
        report = diag.detect_gaps(min_gap=timedelta(hours=12))
        print(f"   [OK] Total gaps: {report.total_gaps}")
        print(f"   [OK] Longest gap: {report.longest_gap}")

        if report.total_gaps > 0:
            print(f"   [OK] Mean gap: {report.gap_summary.get('mean', timedelta(0))}")
            print(
                f"   [OK] Median gap: {report.gap_summary.get('median', timedelta(0))}"
            )

            # Show first 5 gaps
            print("\n   First 5 gaps:")
            for i, (start, end, duration) in enumerate(report.gaps[:5]):
                print(f"      {i + 1}. {start} to {end} ({duration})")

        return True
    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback

        traceback.print_exc()
        return False


def test_calibration_analysis():
    """Test calibration analysis."""
    print("\n" + "=" * 60)
    print("Testing Calibration Analysis")
    print("=" * 60)

    # Load data
    print("\n1. Loading six_hours data...")
    mg = MultiGas(six_hours=str(SIX_HOURS))
    data = mg.six_hours
    print(f"   [OK] Loaded {len(data)} records")

    # Create diagnostics
    diag = DataDiagnostics(data)

    # Analyze calibrations
    print("\n2. Analyzing calibrations...")
    try:
        report = diag.analyze_calibrations()
        print(f"   [OK] Zero calibrations: {report.zero_count}")
        print(f"   [OK] Span calibrations: {report.span_count}")

        if report.calibration_frequency:
            freq_days = report.calibration_frequency.days
            print(f"   [OK] Calibration frequency: {freq_days} days")

        # Show first 5 zero periods
        if report.zero_periods:
            print("\n   First 5 zero calibration periods:")
            for i, (start, end) in enumerate(report.zero_periods[:5]):
                duration = end - start
                print(f"      {i + 1}. {start} to {end} ({duration})")

        return True
    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback

        traceback.print_exc()
        return False


def test_anomaly_detection():
    """Test anomaly detection."""
    print("\n" + "=" * 60)
    print("Testing Anomaly Detection")
    print("=" * 60)

    # Load data
    print("\n1. Loading six_hours data...")
    mg = MultiGas(six_hours=str(SIX_HOURS))
    data = mg.six_hours.filter_column("Status_Flag", "==", 0)  # Only normal data
    print(f"   [OK] Loaded and filtered to {len(data)} records")

    # Create diagnostics
    diag = DataDiagnostics(data)

    # Test different methods
    methods = [
        ("zscore", 3.0),
        ("iqr", 1.5),
        ("mad", 3.0),
    ]

    print("\n2. Testing anomaly detection methods...")
    for method, threshold in methods:
        try:
            report = diag.detect_anomalies(
                "Avg_CO2_lowpass", method=method, threshold=threshold
            )
            pct = (report.anomaly_count / len(data) * 100) if len(data) > 0 else 0
            print(f"   [OK] {method}: {report.anomaly_count} anomalies ({pct:.1f}%)")

            if report.anomaly_count > 0:
                print(
                    f"        Mean: {report.summary_stats.get('mean', 0):.2f}, "
                    f"Max: {report.summary_stats.get('max', 0):.2f}"
                )
        except Exception as e:
            print(f"   [FAIL] {method}: {e}")
            return False

    return True


def test_statistical_summary():
    """Test statistical summary."""
    print("\n" + "=" * 60)
    print("Testing Statistical Summary")
    print("=" * 60)

    # Load data
    print("\n1. Loading six_hours data...")
    mg = MultiGas(six_hours=str(SIX_HOURS))
    data = mg.six_hours
    print(f"   [OK] Loaded {len(data)} records")

    # Create diagnostics
    diag = DataDiagnostics(data)

    # Get summary for gas columns
    print("\n2. Generating statistical summary for gas columns...")
    try:
        gas_cols = ["Avg_CO2_lowpass", "Avg_SO2", "Avg_H2S", "Avg_H2O"]
        summary = diag.get_statistical_summary(columns=gas_cols)

        print(f"   [OK] Summary generated for {len(summary)} columns")

        # Display summary
        print("\n   Statistical Summary:")
        print(
            summary.to_string(
                columns=["column", "count", "mean", "std", "min", "max"],
                index=False,
            )
        )

        # Save summary
        output_path = OUTPUT_DIR / "statistical_summary.csv"
        summary.to_csv(output_path, index=False)
        print(f"\n   [OK] Saved summary to: {output_path}")

        return True
    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback

        traceback.print_exc()
        return False


def test_health_report():
    """Test comprehensive health report."""
    print("\n" + "=" * 60)
    print("Testing Health Report")
    print("=" * 60)

    # Load data
    print("\n1. Loading six_hours data...")
    mg = MultiGas(six_hours=str(SIX_HOURS))
    data = mg.six_hours
    print(f"   [OK] Loaded {len(data)} records")

    # Create diagnostics
    diag = DataDiagnostics(data)

    # Generate health report
    print("\n2. Generating comprehensive health report...")
    try:
        report = diag.generate_health_report()

        print(f"\n   Health Score: {report['health_score']:.1f}/100")

        print("\n   Completeness:")
        print(
            f"      - {report['completeness']['pct']:.1f}% "
            f"({report['completeness']['records']}/{report['completeness']['expected']})"
        )

        print("\n   Gaps:")
        print(f"      - Count: {report['gaps']['count']}")
        print(f"      - Longest: {report['gaps']['longest_hours']:.1f} hours")

        print("\n   Calibrations:")
        print(f"      - Zero: {report['calibrations']['zero_count']}")
        print(f"      - Span: {report['calibrations']['span_count']}")
        if report["calibrations"]["frequency_days"]:
            print(f"      - Frequency: {report['calibrations']['frequency_days']} days")

        print("\n   Recommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"      {i}. {rec}")

        # Save report
        import json

        output_path = OUTPUT_DIR / "health_report.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n   [OK] Saved health report to: {output_path}")

        return True
    except Exception as e:
        print(f"   [FAIL] {e}")
        import traceback

        traceback.print_exc()
        return False


def test_with_two_seconds_data():
    """Test diagnostics with high-frequency two_seconds data."""
    print("\n" + "=" * 60)
    print("Testing with Two Seconds Data")
    print("=" * 60)

    # Load data
    print("\n1. Loading two_seconds data...")
    mg = MultiGas(two_seconds=str(TWO_SECONDS))
    data = mg.two_seconds
    print(f"   [OK] Loaded {len(data)} records")

    # Create diagnostics
    diag = DataDiagnostics(data)

    # Quick completeness check
    print("\n2. Quick completeness analysis...")
    try:
        report = diag.analyze_completeness(expected_freq="2s")
        print(f"   [OK] Completeness: {report.completeness_pct:.1f}%")
        print(f"   [OK] Records: {report.total_records:,}/{report.expected_records:,}")
        print(f"   [OK] Missing periods: {len(report.missing_periods)}")

        return True
    except Exception as e:
        print(f"   [FAIL] {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("magma-multigas v2.0 - Diagnostics Module Test")
    print("=" * 60)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    results = []

    # Run tests
    results.append(("Completeness Analysis", test_completeness_analysis()))
    results.append(("Gap Detection", test_gap_detection()))
    results.append(("Calibration Analysis", test_calibration_analysis()))
    results.append(("Anomaly Detection", test_anomaly_detection()))
    results.append(("Statistical Summary", test_statistical_summary()))
    results.append(("Health Report", test_health_report()))
    results.append(("Two Seconds Data", test_with_two_seconds_data()))

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
