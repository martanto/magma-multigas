#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test markdown export functionality for diagnostics reports."""

import sys
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from magma_multigas import DataDiagnostics, MultiGas

# Data path
DATA_DIR = Path("D:/Data/Multigas/Tangkuban Parahu")
SIX_HOURS = DATA_DIR / "TANG_RTU_Data_6Hr.dat"

# Output directory
OUTPUT_DIR = Path("output/markdown_reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_individual_reports():
    """Test individual report markdown export."""
    print("=" * 70)
    print("Testing Individual Report Markdown Export")
    print("=" * 70)

    mg = MultiGas(six_hours=str(SIX_HOURS))
    diag = DataDiagnostics(mg.six_hours)

    # Test 1: Completeness report
    print("\n1. Completeness Report...")
    completeness = diag.analyze_completeness(expected_freq="6h")
    md_content = completeness.to_markdown()

    output_path = OUTPUT_DIR / "completeness_report.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"   ✓ Saved to: {output_path}")
    print(f"   Size: {len(md_content):,} characters")

    # Test 2: Gap report
    print("\n2. Gap Report...")
    gaps = diag.detect_gaps()
    md_content = gaps.to_markdown()

    output_path = OUTPUT_DIR / "gaps_report.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"   ✓ Saved to: {output_path}")
    print(f"   Size: {len(md_content):,} characters")

    # Test 3: Calibration report
    print("\n3. Calibration Report...")
    calibrations = diag.analyze_calibrations()
    md_content = calibrations.to_markdown()

    output_path = OUTPUT_DIR / "calibrations_report.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"   ✓ Saved to: {output_path}")
    print(f"   Size: {len(md_content):,} characters")

    # Test 4: Anomaly report
    print("\n4. Anomaly Report (Z-Score)...")
    anomalies = diag.detect_anomalies("Avg_CO2_lowpass", method="zscore")
    md_content = anomalies.to_markdown()

    output_path = OUTPUT_DIR / "anomalies_zscore_report.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"   ✓ Saved to: {output_path}")
    print(f"   Size: {len(md_content):,} characters")

    # Test 5: Statistical summary
    print("\n5. Statistical Summary...")
    stats = diag.get_statistical_summary(
        columns=["Avg_CO2_lowpass", "Avg_SO2", "Avg_H2S", "Avg_H2O"]
    )

    from magma_multigas.analysis import markdown as md_utils

    md_content = md_utils.statistics_to_markdown(stats)

    output_path = OUTPUT_DIR / "statistics_report.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"   ✓ Saved to: {output_path}")
    print(f"   Size: {len(md_content):,} characters")

    # Test 6: Health report
    print("\n6. Health Report...")
    health = diag.generate_health_report()
    md_content = md_utils.health_report_to_markdown(health)

    output_path = OUTPUT_DIR / "health_report.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"   ✓ Saved to: {output_path}")
    print(f"   Size: {len(md_content):,} characters")


def test_comprehensive_report():
    """Test comprehensive report generation."""
    print("\n" + "=" * 70)
    print("Testing Comprehensive Report")
    print("=" * 70)

    mg = MultiGas(six_hours=str(SIX_HOURS))
    diag = DataDiagnostics(mg.six_hours)

    print("\nGenerating comprehensive markdown report...")
    output_path = OUTPUT_DIR / "comprehensive_report.md"

    diag.save_report_markdown(
        str(output_path),
        include_completeness=True,
        include_gaps=True,
        include_calibrations=True,
        include_health=True,
    )

    print(f"\n✓ Comprehensive report saved to: {output_path}")

    # Show file size
    file_size = output_path.stat().st_size
    print(f"  Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")


def test_filtered_data_report():
    """Test report generation with filtered data."""
    print("\n" + "=" * 70)
    print("Testing Report with Filtered Data")
    print("=" * 70)

    mg = MultiGas(six_hours=str(SIX_HOURS))

    # Filter to specific time period
    filtered = mg.six_hours.filter_date_range("2024-05-17", "2024-07-24")

    print(f"\nFiltered dataset: {len(filtered)} records")

    diag = DataDiagnostics(filtered)

    print("\nGenerating report for May-July 2024...")
    output_path = OUTPUT_DIR / "filtered_may_july_2024.md"

    diag.save_report_markdown(str(output_path))

    print(f"\n✓ Filtered report saved to: {output_path}")


def test_multiple_anomaly_methods():
    """Test anomaly reports with different methods."""
    print("\n" + "=" * 70)
    print("Testing Multiple Anomaly Detection Methods")
    print("=" * 70)

    mg = MultiGas(six_hours=str(SIX_HOURS))
    data = mg.six_hours.filter_column("Status_Flag", "==", 0)
    diag = DataDiagnostics(data)

    methods = [
        ("zscore", 3.0, "Z-Score"),
        ("iqr", 1.5, "IQR"),
        ("mad", 3.0, "MAD"),
    ]

    print("\nGenerating anomaly reports for Avg_CO2_lowpass...")

    for method, threshold, name in methods:
        report = diag.detect_anomalies(
            "Avg_CO2_lowpass", method=method, threshold=threshold
        )

        md_content = report.to_markdown()
        output_path = OUTPUT_DIR / f"anomalies_{method}.md"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"  ✓ {name:10s} → {output_path.name} ({report.anomaly_count} anomalies)")


def preview_report():
    """Preview a sample report."""
    print("\n" + "=" * 70)
    print("Preview: Health Report")
    print("=" * 70)

    mg = MultiGas(six_hours=str(SIX_HOURS))
    diag = DataDiagnostics(mg.six_hours)

    health = diag.generate_health_report()

    from magma_multigas.analysis import markdown as md_utils

    md_content = md_utils.health_report_to_markdown(health)

    # Print first 40 lines
    lines = md_content.split("\n")
    print("\n" + "\n".join(lines[:40]))

    if len(lines) > 40:
        print(f"\n... ({len(lines) - 40} more lines)")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("MARKDOWN EXPORT FUNCTIONALITY TEST")
    print("=" * 70)
    print(f"\nData: {SIX_HOURS}")
    print(f"Output: {OUTPUT_DIR}\n")

    try:
        # Run tests
        test_individual_reports()
        test_comprehensive_report()
        test_filtered_data_report()
        test_multiple_anomaly_methods()
        preview_report()

        # Summary
        print("\n" + "=" * 70)
        print("TEST COMPLETE")
        print("=" * 70)

        # List generated files
        print(f"\n📄 Generated Reports ({OUTPUT_DIR}):\n")
        for file_path in sorted(OUTPUT_DIR.glob("*.md")):
            size = file_path.stat().st_size
            print(f"   {file_path.name:40s} {size:8,} bytes")

        print(f"\n✅ All markdown reports generated successfully!")
        print(f"\n💡 Tip: Open the markdown files in a markdown viewer or editor")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
