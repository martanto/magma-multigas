"""Markdown export utilities for diagnostics reports."""

from datetime import timedelta
from typing import Any, Dict, List

import pandas as pd


def format_timedelta(td: timedelta) -> str:
    """Format timedelta as human-readable string.

    Args:
        td: Timedelta to format

    Returns:
        Formatted string (e.g., "5 days 3 hours")
    """
    days = td.days
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60

    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0 and days == 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")

    return " ".join(parts) if parts else "0 minutes"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage with visual bar.

    Args:
        value: Percentage value (0-100)
        decimals: Number of decimal places

    Returns:
        Formatted string with bar
    """
    bar_length = 20
    filled = int(value / 100 * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    return f"`{bar}` {value:.{decimals}f}%"


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = None) -> str:
    """Convert DataFrame to markdown table.

    Args:
        df: DataFrame to convert
        max_rows: Maximum number of rows to include

    Returns:
        Markdown table string
    """
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)

    return df.to_markdown(index=False)


def completeness_to_markdown(
    total_records: int,
    expected_records: int,
    completeness_pct: float,
    date_range: tuple,
    column_stats: pd.DataFrame,
    missing_periods: List[tuple],
) -> str:
    """Generate markdown report for completeness analysis.

    Args:
        total_records: Total number of records
        expected_records: Expected number of records
        completeness_pct: Completeness percentage
        date_range: Tuple of (start_date, end_date)
        column_stats: DataFrame with per-column statistics
        missing_periods: List of (start, end) tuples for missing periods

    Returns:
        Markdown formatted report
    """
    md = []

    # Header
    md.append("# Data Completeness Report\n")

    # Overall summary
    md.append("## Overall Completeness\n")
    md.append(f"- **Completeness**: {format_percentage(completeness_pct)}")
    md.append(f"- **Records**: {total_records:,} / {expected_records:,} expected")
    md.append(
        f"- **Date Range**: {date_range[0].strftime('%Y-%m-%d')} to "
        f"{date_range[1].strftime('%Y-%m-%d')}"
    )
    duration = (date_range[1] - date_range[0]).days
    md.append(f"- **Duration**: {duration:,} days\n")

    # Missing periods
    if missing_periods:
        md.append(f"## Missing Periods ({len(missing_periods)})\n")
        for i, (start, end) in enumerate(missing_periods[:10], 1):
            duration = format_timedelta(end - start)
            md.append(
                f"{i}. **{start.strftime('%Y-%m-%d %H:%M')}** to "
                f"**{end.strftime('%Y-%m-%d %H:%M')}** ({duration})"
            )
        if len(missing_periods) > 10:
            md.append(f"\n*...and {len(missing_periods) - 10} more*")
        md.append("")

    # Top complete columns
    md.append("## Most Complete Columns (Top 10)\n")
    top_cols = column_stats.head(10).copy()
    top_cols["completeness"] = top_cols["completeness_pct"].apply(
        lambda x: format_percentage(x)
    )
    md.append(
        top_cols[["column", "completeness", "available", "total"]].to_markdown(
            index=False
        )
    )
    md.append("")

    # Incomplete columns
    incomplete = column_stats[column_stats["completeness_pct"] < 100]
    if len(incomplete) > 0:
        md.append(f"## Incomplete Columns ({len(incomplete)})\n")
        inc_cols = incomplete.copy()
        inc_cols["completeness"] = inc_cols["completeness_pct"].apply(
            lambda x: format_percentage(x)
        )
        md.append(
            inc_cols[["column", "completeness", "missing"]].to_markdown(index=False)
        )
        md.append("")

    return "\n".join(md)


def gaps_to_markdown(
    total_gaps: int, longest_gap: timedelta, gaps: List[tuple], gap_summary: Dict
) -> str:
    """Generate markdown report for gap analysis.

    Args:
        total_gaps: Total number of gaps
        longest_gap: Longest gap duration
        gaps: List of (start, end, duration) tuples
        gap_summary: Dictionary with gap statistics

    Returns:
        Markdown formatted report
    """
    md = []

    # Header
    md.append("# Data Gap Analysis Report\n")

    # Summary
    md.append("## Summary\n")
    md.append(f"- **Total Gaps**: {total_gaps}")
    if total_gaps > 0:
        md.append(f"- **Longest Gap**: {format_timedelta(longest_gap)}")
        md.append(
            f"- **Mean Gap**: {format_timedelta(gap_summary.get('mean', timedelta()))}"
        )
        median_gap = format_timedelta(gap_summary.get("median", timedelta()))
        md.append(f"- **Median Gap**: {median_gap}")
        md.append("")

        # Gap details
        md.append("## Gap Details\n")
        md.append("| # | Start | End | Duration | Hours |")
        md.append("|---|-------|-----|----------|-------|")

        for i, (start, end, duration) in enumerate(gaps[:20], 1):
            hours = duration.total_seconds() / 3600
            md.append(
                f"| {i} | {start.strftime('%Y-%m-%d %H:%M')} | "
                f"{end.strftime('%Y-%m-%d %H:%M')} | "
                f"{format_timedelta(duration)} | {hours:.1f} |"
            )

        if len(gaps) > 20:
            md.append(f"\n*...and {len(gaps) - 20} more gaps*")
    else:
        md.append("\n✅ No gaps detected!")

    md.append("")
    return "\n".join(md)


def calibrations_to_markdown(
    zero_periods: List[tuple],
    span_periods: List[tuple],
    zero_count: int,
    span_count: int,
    calibration_frequency: timedelta,
) -> str:
    """Generate markdown report for calibration analysis.

    Args:
        zero_periods: List of (start, end) tuples for zero calibrations
        span_periods: List of (start, end) tuples for span calibrations
        zero_count: Total zero calibrations
        span_count: Total span calibrations
        calibration_frequency: Average time between calibrations

    Returns:
        Markdown formatted report
    """
    md = []

    # Header
    md.append("# Calibration Analysis Report\n")

    # Summary
    md.append("## Summary\n")
    md.append(f"- **Zero Calibrations**: {zero_count}")
    md.append(f"- **Span Calibrations**: {span_count}")
    if calibration_frequency:
        freq_str = format_timedelta(calibration_frequency)
        md.append(f"- **Calibration Frequency**: Every {freq_str}")
    md.append("")

    # Zero calibrations
    if zero_periods:
        md.append(f"## Zero Calibration Periods ({len(zero_periods)})\n")
        md.append("| # | Start | End | Duration |")
        md.append("|---|-------|-----|----------|")

        for i, (start, end) in enumerate(zero_periods[:20], 1):
            duration = format_timedelta(end - start)
            md.append(
                f"| {i} | {start.strftime('%Y-%m-%d %H:%M')} | "
                f"{end.strftime('%Y-%m-%d %H:%M')} | {duration} |"
            )

        if len(zero_periods) > 20:
            md.append(f"\n*...and {len(zero_periods) - 20} more*")
        md.append("")

    # Span calibrations
    if span_periods:
        md.append(f"## Span Calibration Periods ({len(span_periods)})\n")
        md.append("| # | Start | End | Duration |")
        md.append("|---|-------|-----|----------|")

        for i, (start, end) in enumerate(span_periods[:20], 1):
            duration = format_timedelta(end - start)
            md.append(
                f"| {i} | {start.strftime('%Y-%m-%d %H:%M')} | "
                f"{end.strftime('%Y-%m-%d %H:%M')} | {duration} |"
            )

        if len(span_periods) > 20:
            md.append(f"\n*...and {len(span_periods) - 20} more*")
        md.append("")

    if zero_count == 0 and span_count == 0:
        md.append("⚠️ **Warning**: No calibration periods detected\n")
        md.append("This may indicate:")
        md.append("- Status_Flag column not populated")
        md.append("- No calibrations performed during this period")
        md.append("- Different status flag values used")
        md.append("")

    return "\n".join(md)


def anomalies_to_markdown(
    column: str,
    anomaly_count: int,
    method: str,
    threshold: float,
    summary_stats: Dict[str, float],
    total_records: int,
) -> str:
    """Generate markdown report for anomaly detection.

    Args:
        column: Column analyzed
        anomaly_count: Number of anomalies detected
        method: Detection method used
        threshold: Threshold value
        summary_stats: Dictionary with anomaly statistics
        total_records: Total number of records analyzed

    Returns:
        Markdown formatted report
    """
    md = []

    # Header
    md.append(f"# Anomaly Detection Report: {column}\n")

    # Summary
    md.append("## Summary\n")
    md.append(f"- **Method**: {method.upper()}")
    md.append(f"- **Threshold**: {threshold}")
    md.append(f"- **Total Records**: {total_records:,}")
    md.append(f"- **Anomalies Detected**: {anomaly_count}")

    if total_records > 0:
        pct = anomaly_count / total_records * 100
        md.append(f"- **Percentage**: {format_percentage(pct)}")

    md.append("")

    # Statistics
    if anomaly_count > 0 and summary_stats:
        md.append("## Anomaly Statistics\n")
        md.append("| Statistic | Value |")
        md.append("|-----------|-------|")
        md.append(f"| Count | {summary_stats.get('count', 0)} |")
        md.append(f"| Mean | {summary_stats.get('mean', 0):.2f} |")
        md.append(f"| Std Dev | {summary_stats.get('std', 0):.2f} |")
        md.append(f"| Min | {summary_stats.get('min', 0):.2f} |")
        md.append(f"| Max | {summary_stats.get('max', 0):.2f} |")
        md.append("")
    else:
        md.append("✅ No anomalies detected with current settings.\n")

    # Method explanation
    md.append("## Detection Method\n")
    if method == "zscore":
        md.append(
            f"**Z-Score Method**: Detects values more than {threshold} standard "
            "deviations from the mean. Best for normally distributed data."
        )
    elif method == "iqr":
        md.append(
            f"**IQR Method**: Detects values outside Q1 - {threshold}×IQR or "
            f"Q3 + {threshold}×IQR. Robust to outliers."
        )
    elif method == "mad":
        md.append(
            f"**MAD Method**: Uses median absolute deviation with "
            f"threshold {threshold}. Very robust for non-normal distributions."
        )

    md.append("")
    return "\n".join(md)


def statistics_to_markdown(summary: pd.DataFrame) -> str:
    """Generate markdown report for statistical summary.

    Args:
        summary: DataFrame with statistical summary

    Returns:
        Markdown formatted report
    """
    md = []

    # Header
    md.append("# Statistical Summary Report\n")

    # Table
    md.append("## Column Statistics\n")

    # Format for better readability
    display_cols = ["column", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    available_cols = [col for col in display_cols if col in summary.columns]

    md.append(summary[available_cols].to_markdown(index=False, floatfmt=".2f"))
    md.append("")

    # Additional metrics if available
    if "skewness" in summary.columns or "kurtosis" in summary.columns:
        md.append("## Distribution Metrics\n")
        dist_cols = ["column"]
        if "skewness" in summary.columns:
            dist_cols.append("skewness")
        if "kurtosis" in summary.columns:
            dist_cols.append("kurtosis")
        if "missing_pct" in summary.columns:
            dist_cols.append("missing_pct")

        md.append(summary[dist_cols].to_markdown(index=False, floatfmt=".2f"))
        md.append("")

    return "\n".join(md)


def health_report_to_markdown(health: Dict[str, Any]) -> str:
    """Generate markdown report for health assessment.

    Args:
        health: Health report dictionary

    Returns:
        Markdown formatted report
    """
    md = []

    # Header
    score = health["health_score"]
    if score >= 80:
        status = "🟢 EXCELLENT"
    elif score >= 60:
        status = "🟡 GOOD"
    elif score >= 40:
        status = "🟠 FAIR"
    else:
        status = "🔴 POOR"

    md.append(f"# Data Health Report: {status}\n")

    # Overall score
    md.append(f"## Overall Health Score: {score:.1f}/100\n")
    md.append(format_percentage(score))
    md.append("")

    # Detailed metrics
    md.append("## Detailed Metrics\n")

    # Completeness
    comp = health["completeness"]
    md.append("### Completeness")
    md.append(f"- **Percentage**: {format_percentage(comp['pct'])}")
    md.append(f"- **Records**: {comp['records']:,} / {comp['expected']:,}")
    md.append("")

    # Gaps
    gaps = health["gaps"]
    md.append("### Data Gaps")
    md.append(f"- **Count**: {gaps['count']}")
    md.append(f"- **Longest**: {gaps['longest_hours']:.1f} hours")
    md.append("")

    # Calibrations
    cal = health["calibrations"]
    md.append("### Calibrations")
    md.append(f"- **Zero**: {cal['zero_count']}")
    md.append(f"- **Span**: {cal['span_count']}")
    if cal["frequency_days"]:
        md.append(f"- **Frequency**: Every {cal['frequency_days']} days")
    md.append("")

    # Recommendations
    md.append("## Recommendations\n")
    if health["recommendations"]:
        for i, rec in enumerate(health["recommendations"], 1):
            md.append(f"{i}. {rec}")
    else:
        md.append("✅ No issues detected. Data quality is excellent!")

    md.append("")

    # Score breakdown
    md.append("## Score Breakdown\n")
    md.append("| Component | Weight | Score |")
    md.append("|-----------|--------|-------|")
    md.append(f"| Completeness | 40% | {comp['pct'] * 0.4:.1f} points |")

    gap_score = max(0, 30 - min(30, gaps["count"] + gaps["longest_hours"] / 24))
    md.append(f"| Gaps | 30% | {gap_score:.1f} points |")

    cal_score = 30 if cal["zero_count"] > 0 else 0
    md.append(f"| Calibrations | 30% | {cal_score:.1f} points |")
    md.append(f"| **Total** | **100%** | **{score:.1f} points** |")
    md.append("")

    return "\n".join(md)
