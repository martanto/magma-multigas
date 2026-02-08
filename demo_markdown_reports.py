#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Interactive demonstration of markdown report generation."""

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
OUTPUT_DIR = Path("output/demo_reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_file_preview(file_path, max_lines=30):
    """Print preview of markdown file."""
    print(f"\n📄 Preview: {file_path.name}")
    print("-" * 70)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines[:max_lines], 1):
        print(f"{i:3d} | {line.rstrip()}")

    if len(lines) > max_lines:
        print(f"\n... ({len(lines) - max_lines} more lines)")

    print("-" * 70)
    print(f"Total: {len(lines)} lines, {file_path.stat().st_size:,} bytes\n")


def demo_quick_health_report():
    """Quick health report generation."""
    print_header("1. QUICK HEALTH REPORT")

    print("\n📋 Generating health report...")

    mg = MultiGas(six_hours=str(SIX_HOURS))
    diag = DataDiagnostics(mg.six_hours)

    # Generate health report
    health = diag.generate_health_report()

    # Export to markdown
    from magma_multigas.analysis import markdown as md_utils

    markdown_text = md_utils.health_report_to_markdown(health)

    # Save to file
    output_path = OUTPUT_DIR / "quick_health_report.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    print(f"✓ Saved to: {output_path}")
    print(f"  Size: {len(markdown_text):,} characters")

    # Preview
    print_file_preview(output_path, max_lines=35)


def demo_comprehensive_report():
    """Generate comprehensive report with all sections."""
    print_header("2. COMPREHENSIVE REPORT")

    print("\n📊 Generating comprehensive report with all sections...")

    mg = MultiGas(six_hours=str(SIX_HOURS))
    diag = DataDiagnostics(mg.six_hours)

    # Generate comprehensive report
    output_path = OUTPUT_DIR / "comprehensive_report.md"
    diag.save_report_markdown(
        str(output_path),
        include_completeness=True,
        include_gaps=True,
        include_calibrations=True,
        include_health=True,
    )

    file_size = output_path.stat().st_size
    print(f"✓ Saved to: {output_path}")
    print(f"  Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

    # Preview first part
    print_file_preview(output_path, max_lines=50)


def demo_custom_report():
    """Generate custom report with selected sections."""
    print_header("3. CUSTOM REPORT (Selected Sections)")

    print("\n🔧 Generating custom report (health + completeness only)...")

    mg = MultiGas(six_hours=str(SIX_HOURS))
    diag = DataDiagnostics(mg.six_hours)

    # Generate custom report
    output_path = OUTPUT_DIR / "custom_report.md"
    diag.save_report_markdown(
        str(output_path),
        include_completeness=True,
        include_gaps=False,  # Skip gaps
        include_calibrations=False,  # Skip calibrations
        include_health=True,
    )

    file_size = output_path.stat().st_size
    print(f"✓ Saved to: {output_path}")
    print(f"  Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print(f"  Sections: Health + Completeness only")


def demo_filtered_period_report():
    """Generate report for filtered time period."""
    print_header("4. FILTERED PERIOD REPORT")

    print("\n📅 Generating report for May-July 2024...")

    mg = MultiGas(six_hours=str(SIX_HOURS))

    # Filter to specific period
    filtered = mg.six_hours.filter_date_range("2024-05-17", "2024-07-24")
    print(f"  Filtered to {len(filtered)} records")

    diag = DataDiagnostics(filtered)

    # Generate report for this period
    output_path = OUTPUT_DIR / "may_july_2024_report.md"
    diag.save_report_markdown(str(output_path))

    file_size = output_path.stat().st_size
    print(f"✓ Saved to: {output_path}")
    print(f"  Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print(f"  Period: May 17 - July 24, 2024")


def demo_individual_reports():
    """Generate individual report types."""
    print_header("5. INDIVIDUAL REPORT TYPES")

    mg = MultiGas(six_hours=str(SIX_HOURS))
    diag = DataDiagnostics(mg.six_hours)

    reports = []

    # Completeness report
    print("\n📊 Completeness Report...")
    completeness = diag.analyze_completeness(expected_freq="6h")
    md_text = completeness.to_markdown()
    output_path = OUTPUT_DIR / "individual_completeness.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    reports.append(("Completeness", output_path, len(md_text)))
    print(f"  ✓ {output_path.name} ({len(md_text):,} chars)")

    # Gap report
    print("\n🕳️  Gap Report...")
    gaps = diag.detect_gaps()
    md_text = gaps.to_markdown()
    output_path = OUTPUT_DIR / "individual_gaps.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    reports.append(("Gaps", output_path, len(md_text)))
    print(f"  ✓ {output_path.name} ({len(md_text):,} chars)")

    # Calibration report
    print("\n🔧 Calibration Report...")
    calibrations = diag.analyze_calibrations()
    md_text = calibrations.to_markdown()
    output_path = OUTPUT_DIR / "individual_calibrations.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    reports.append(("Calibrations", output_path, len(md_text)))
    print(f"  ✓ {output_path.name} ({len(md_text):,} chars)")

    # Anomaly report
    print("\n🔍 Anomaly Report (Z-Score)...")
    anomalies = diag.detect_anomalies("Avg_CO2_lowpass", method="zscore")
    md_text = anomalies.to_markdown()
    output_path = OUTPUT_DIR / "individual_anomalies.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    reports.append(("Anomalies", output_path, len(md_text)))
    print(f"  ✓ {output_path.name} ({len(md_text):,} chars)")

    # Statistical summary
    print("\n📈 Statistical Summary...")
    stats = diag.get_statistical_summary(
        columns=["Avg_CO2_lowpass", "Avg_SO2", "Avg_H2S", "Avg_H2O"]
    )

    from magma_multigas.analysis import markdown as md_utils

    md_text = md_utils.statistics_to_markdown(stats)
    output_path = OUTPUT_DIR / "individual_statistics.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    reports.append(("Statistics", output_path, len(md_text)))
    print(f"  ✓ {output_path.name} ({len(md_text):,} chars)")

    print(f"\n✓ Generated {len(reports)} individual reports")


def demo_multiple_stations():
    """Simulate multiple station reports."""
    print_header("6. BATCH PROCESSING SIMULATION")

    print("\n🏭 Generating reports for different time periods...")
    print("   (Simulating multiple stations)\n")

    mg = MultiGas(six_hours=str(SIX_HOURS))

    # Different time periods to simulate different stations
    periods = [
        ("May 2024", "2024-05-01", "2024-05-31"),
        ("June 2024", "2024-06-01", "2024-06-30"),
        ("July 2024", "2024-07-01", "2024-07-31"),
    ]

    for period_name, start, end in periods:
        filtered = mg.six_hours.filter_date_range(start, end)

        if len(filtered) > 0:
            diag = DataDiagnostics(filtered)

            # Generate report
            output_path = OUTPUT_DIR / f"period_{period_name.replace(' ', '_')}.md"
            diag.save_report_markdown(str(output_path))

            health = diag.generate_health_report()
            file_size = output_path.stat().st_size

            print(
                f"  ✓ {period_name:12s} → {output_path.name:30s} "
                f"({file_size:5,} bytes, health: {health['health_score']:.0f}/100)"
            )


def demo_comparison_table():
    """Create markdown comparison table."""
    print_header("7. COMPARISON TABLE")

    print("\n📊 Creating comparison table across periods...\n")

    mg = MultiGas(six_hours=str(SIX_HOURS))

    periods = [
        ("May 2024", "2024-05-01", "2024-05-31"),
        ("June 2024", "2024-06-01", "2024-06-30"),
        ("July 2024", "2024-07-01", "2024-07-31"),
    ]

    # Collect data
    comparison_data = []
    for period_name, start, end in periods:
        filtered = mg.six_hours.filter_date_range(start, end)

        if len(filtered) > 0:
            diag = DataDiagnostics(filtered)
            health = diag.generate_health_report()
            gaps = diag.detect_gaps()

            comparison_data.append(
                {
                    "period": period_name,
                    "records": len(filtered),
                    "completeness": health["completeness"]["pct"],
                    "gaps": gaps.total_gaps,
                    "health_score": health["health_score"],
                }
            )

    # Create markdown table
    md_lines = []
    md_lines.append("# Period Comparison Report\n")
    md_lines.append("## Summary Table\n")
    md_lines.append("| Period | Records | Completeness | Gaps | Health Score |")
    md_lines.append("|--------|---------|--------------|------|--------------|")

    for data in comparison_data:
        md_lines.append(
            f"| {data['period']} | {data['records']:,} | "
            f"{data['completeness']:.1f}% | {data['gaps']} | "
            f"{data['health_score']:.1f}/100 |"
        )

    md_lines.append("")

    # Add summary
    avg_health = sum(d["health_score"] for d in comparison_data) / len(
        comparison_data
    )
    total_records = sum(d["records"] for d in comparison_data)

    md_lines.append("## Summary Statistics\n")
    md_lines.append(f"- **Total Records**: {total_records:,}")
    md_lines.append(f"- **Average Health Score**: {avg_health:.1f}/100")
    md_lines.append(f"- **Periods Analyzed**: {len(comparison_data)}")

    # Save
    output_path = OUTPUT_DIR / "comparison_table.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"✓ Saved to: {output_path}")
    print_file_preview(output_path)


def demo_usage_examples():
    """Show code examples."""
    print_header("8. USAGE EXAMPLES")

    print("\n💡 How to use markdown export in your code:\n")

    examples = [
        (
            "Quick Health Report",
            """from magma_multigas import MultiGas, DataDiagnostics
from magma_multigas.analysis import markdown as md_utils

mg = MultiGas(six_hours="data.dat")
diag = DataDiagnostics(mg.six_hours)

# Generate and export health report
health = diag.generate_health_report()
markdown = md_utils.health_report_to_markdown(health)

with open("health.md", "w", encoding="utf-8") as f:
    f.write(markdown)""",
        ),
        (
            "Individual Report",
            """# Generate completeness report
report = diag.analyze_completeness(expected_freq="6h")

# Export to markdown
markdown = report.to_markdown()

# Save to file
with open("completeness.md", "w", encoding="utf-8") as f:
    f.write(markdown)""",
        ),
        (
            "Comprehensive Report",
            """# One-liner for complete report
diag.save_report_markdown(
    "full_report.md",
    include_completeness=True,
    include_gaps=True,
    include_calibrations=True,
    include_health=True
)""",
        ),
        (
            "Custom Sections",
            """# Generate report with selected sections only
diag.save_report_markdown(
    "custom_report.md",
    include_completeness=True,  # Include
    include_gaps=False,         # Skip
    include_calibrations=False, # Skip
    include_health=True         # Include
)""",
        ),
    ]

    for i, (title, code) in enumerate(examples, 1):
        print(f"{i}. {title}:")
        print()
        for line in code.split("\n"):
            print(f"   {line}")
        print()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  MARKDOWN REPORT GENERATION - INTERACTIVE DEMO")
    print("=" * 70)
    print(f"\n  Dataset: {SIX_HOURS.name}")
    print(f"  Output: {OUTPUT_DIR}\n")

    try:
        # Run demonstrations
        demo_quick_health_report()
        demo_comprehensive_report()
        demo_custom_report()
        demo_filtered_period_report()
        demo_individual_reports()
        demo_multiple_stations()
        demo_comparison_table()
        demo_usage_examples()

        # Final summary
        print_header("DEMONSTRATION COMPLETE")

        # List all generated files
        print(f"\n📁 Generated Reports ({OUTPUT_DIR}):\n")

        total_size = 0
        files = sorted(OUTPUT_DIR.glob("*.md"))

        for file_path in files:
            size = file_path.stat().st_size
            total_size += size
            print(f"   {file_path.name:40s} {size:8,} bytes")

        print(f"\n   {'TOTAL':40s} {total_size:8,} bytes ({total_size/1024:.1f} KB)")

        print("\n" + "=" * 70)
        print("✅ All markdown reports generated successfully!")
        print("=" * 70)

        print("\n💡 Tips:")
        print("  • Open .md files in VS Code, GitHub, or any markdown viewer")
        print("  • Convert to PDF: pandoc report.md -o report.pdf")
        print("  • Convert to HTML: pandoc report.md -o report.html")
        print("  • Share via email, Slack, or attach to GitHub issues")
        print("  • Version control friendly (plain text, git diff works)")

        print("\n📚 Documentation:")
        print("  • PHASE4_ANALYSIS_SUMMARY.md - Full technical docs")
        print("  • DIAGNOSTICS_QUICK_START.md - Quick reference")
        print("  • test_markdown_export.py - Test examples")

        print()

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
