"""Data quality diagnostics and analysis tools.

This module provides comprehensive data quality analysis for volcanic gas
monitoring datasets, including completeness checks, gap detection, anomaly
identification, and statistical summaries.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config.logging import get_logger
from ..core.types import ColumnName
from ..data.dataset import Dataset

logger = get_logger(__name__)


@dataclass
class CompletenessReport:
    """Data completeness analysis results.

    Attributes:
        total_records: Total number of records in dataset
        date_range: Tuple of (start_date, end_date)
        expected_records: Expected number of records based on frequency
        completeness_pct: Overall completeness percentage
        column_stats: Per-column completeness statistics
        missing_periods: List of time periods with no data
    """

    total_records: int
    date_range: Tuple[pd.Timestamp, pd.Timestamp]
    expected_records: int
    completeness_pct: float
    column_stats: pd.DataFrame
    missing_periods: List[Tuple[pd.Timestamp, pd.Timestamp]]


@dataclass
class GapReport:
    """Data gap analysis results.

    Attributes:
        total_gaps: Total number of gaps found
        longest_gap: Longest gap duration
        gaps: List of (start, end, duration) tuples
        gap_summary: Summary statistics of gaps
    """

    total_gaps: int
    longest_gap: timedelta
    gaps: List[Tuple[pd.Timestamp, pd.Timestamp, timedelta]]
    gap_summary: Dict[str, Any]


@dataclass
class CalibrationReport:
    """Calibration period analysis results.

    Attributes:
        zero_periods: List of zero calibration periods
        span_periods: List of span calibration periods
        zero_count: Total number of zero calibrations
        span_count: Total number of span calibrations
        calibration_frequency: Average time between calibrations
    """

    zero_periods: List[Tuple[pd.Timestamp, pd.Timestamp]]
    span_periods: List[Tuple[pd.Timestamp, pd.Timestamp]]
    zero_count: int
    span_count: int
    calibration_frequency: Optional[timedelta]


@dataclass
class AnomalyReport:
    """Anomaly detection results.

    Attributes:
        column: Column analyzed
        anomaly_count: Number of anomalies detected
        anomaly_indices: Indices of anomalous values
        method: Detection method used
        threshold: Threshold used for detection
        summary_stats: Statistical summary of anomalies
    """

    column: str
    anomaly_count: int
    anomaly_indices: List[int]
    method: str
    threshold: float
    summary_stats: Dict[str, float]


class DataDiagnostics:
    """Comprehensive data quality diagnostics for volcanic gas monitoring data.

    This class provides various methods to analyze data quality, detect issues,
    and generate reports. All methods work with Dataset objects and return
    structured report objects.

    Example:
        >>> from magma_multigas import MultiGas, DataDiagnostics
        >>> mg = MultiGas(six_hours="data.dat")
        >>> diag = DataDiagnostics(mg.six_hours)
        >>> completeness = diag.analyze_completeness()
        >>> print(f"Overall completeness: {completeness.completeness_pct:.1f}%")
    """

    def __init__(self, dataset: Dataset):
        """Initialize diagnostics with a dataset.

        Args:
            dataset: Dataset to analyze
        """
        self.dataset = dataset
        self.df = dataset.df
        logger.debug(f"Initialized diagnostics for {len(dataset)} records")

    def analyze_completeness(
        self, expected_freq: Optional[str] = None
    ) -> CompletenessReport:
        """Analyze data completeness.

        Calculates overall and per-column completeness statistics, identifies
        missing data periods, and compares actual vs expected record counts.

        Args:
            expected_freq: Expected data frequency (e.g., "6h", "2s", "1min")
                          If None, infers from data

        Returns:
            CompletenessReport with detailed completeness statistics

        Example:
            >>> report = diag.analyze_completeness(expected_freq="6h")
            >>> print(f"Found {len(report.missing_periods)} missing periods")
        """
        logger.info("Analyzing data completeness...")

        # Basic stats
        total_records = len(self.df)
        date_range = (self.df.index.min(), self.df.index.max())

        # Infer frequency if not provided
        if expected_freq is None:
            expected_freq = pd.infer_freq(self.df.index)
            logger.debug(f"Inferred frequency: {expected_freq}")

        # Calculate expected records
        if expected_freq:
            expected_records = len(
                pd.date_range(
                    start=date_range[0], end=date_range[1], freq=expected_freq
                )
            )
        else:
            expected_records = total_records
            logger.warning("Could not infer frequency, using actual record count")

        completeness_pct = (
            (total_records / expected_records * 100) if expected_records > 0 else 0
        )

        # Per-column statistics
        column_stats = []
        for col in self.df.columns:
            non_null = self.df[col].notna().sum()
            column_stats.append(
                {
                    "column": col,
                    "total": total_records,
                    "available": non_null,
                    "missing": total_records - non_null,
                    "completeness_pct": (non_null / total_records * 100)
                    if total_records > 0
                    else 0,
                }
            )

        column_stats_df = pd.DataFrame(column_stats).sort_values(
            "completeness_pct", ascending=False
        )

        # Find missing periods
        missing_periods = self._find_missing_periods(expected_freq)

        logger.info(
            f"Completeness: {completeness_pct:.1f}% "
            f"({total_records}/{expected_records} records)"
        )

        return CompletenessReport(
            total_records=total_records,
            date_range=date_range,
            expected_records=expected_records,
            completeness_pct=completeness_pct,
            column_stats=column_stats_df,
            missing_periods=missing_periods,
        )

    def detect_gaps(
        self, max_gap: Optional[timedelta] = None, min_gap: Optional[timedelta] = None
    ) -> GapReport:
        """Detect gaps in time series data.

        Identifies periods where data is missing or sampling frequency is
        interrupted. Useful for finding data transmission issues or sensor
        failures.

        Args:
            max_gap: Maximum expected gap between records. If None, uses 2x
                    the median interval
            min_gap: Minimum gap duration to report. If None, reports all gaps

        Returns:
            GapReport with gap locations and statistics

        Example:
            >>> from datetime import timedelta
            >>> report = diag.detect_gaps(min_gap=timedelta(hours=12))
            >>> print(f"Found {report.total_gaps} gaps > 12 hours")
        """
        logger.info("Detecting data gaps...")

        # Calculate time differences
        time_diffs = self.df.index.to_series().diff()

        # Determine gap threshold
        if max_gap is None:
            median_interval = time_diffs.median()
            max_gap = median_interval * 2
            logger.debug(f"Using auto gap threshold: {max_gap}")

        # Find gaps
        gaps_mask = time_diffs > max_gap
        gap_starts = self.df.index[gaps_mask]

        gaps = []
        for gap_start in gap_starts:
            gap_idx = self.df.index.get_loc(gap_start)
            gap_prev = self.df.index[gap_idx - 1]
            duration = gap_start - gap_prev

            # Apply minimum gap filter
            if min_gap is None or duration >= min_gap:
                gaps.append((gap_prev, gap_start, duration))

        # Calculate statistics
        if gaps:
            durations = [gap[2] for gap in gaps]
            longest_gap = max(durations)
            gap_summary = {
                "count": len(gaps),
                "longest": longest_gap,
                "shortest": min(durations),
                "mean": sum(durations, timedelta()) / len(durations),
                "median": sorted(durations)[len(durations) // 2],
            }
        else:
            longest_gap = timedelta(0)
            gap_summary = {"count": 0}

        logger.info(f"Found {len(gaps)} gaps (longest: {longest_gap})")

        return GapReport(
            total_gaps=len(gaps),
            longest_gap=longest_gap,
            gaps=gaps,
            gap_summary=gap_summary,
        )

    def analyze_calibrations(
        self, status_column: str = "Status_Flag"
    ) -> CalibrationReport:
        """Analyze calibration periods.

        Identifies zero and span calibration periods based on status flags
        and calculates calibration frequency.

        Args:
            status_column: Column containing status flags
                          (0=normal, 1=zero, 2=span)

        Returns:
            CalibrationReport with calibration period information

        Example:
            >>> report = diag.analyze_calibrations()
            >>> print(f"Zero calibrations: {report.zero_count}")
            >>> print(f"Span calibrations: {report.span_count}")
        """
        logger.info("Analyzing calibration periods...")

        if status_column not in self.df.columns:
            logger.warning(f"Column '{status_column}' not found")
            return CalibrationReport(
                zero_periods=[],
                span_periods=[],
                zero_count=0,
                span_count=0,
                calibration_frequency=None,
            )

        # Find calibration periods
        zero_periods = self._find_periods(self.df, status_column, 1)
        span_periods = self._find_periods(self.df, status_column, 2)

        # Calculate calibration frequency
        all_cal_starts = [p[0] for p in zero_periods + span_periods]
        if len(all_cal_starts) > 1:
            all_cal_starts.sort()
            intervals = [
                all_cal_starts[i + 1] - all_cal_starts[i]
                for i in range(len(all_cal_starts) - 1)
            ]
            avg_interval = sum(intervals, timedelta()) / len(intervals)
        else:
            avg_interval = None

        logger.info(
            f"Found {len(zero_periods)} zero and {len(span_periods)} span calibrations"
        )

        return CalibrationReport(
            zero_periods=zero_periods,
            span_periods=span_periods,
            zero_count=len(zero_periods),
            span_count=len(span_periods),
            calibration_frequency=avg_interval,
        )

    def detect_anomalies(
        self,
        column: ColumnName,
        method: str = "zscore",
        threshold: float = 3.0,
        ignore_nulls: bool = True,
    ) -> AnomalyReport:
        """Detect anomalies in a data column.

        Identifies unusual values using statistical methods. Useful for
        finding sensor errors, data transmission issues, or unusual events.

        Args:
            column: Column to analyze
            method: Detection method ("zscore", "iqr", "mad")
                   - zscore: Standard deviation based (default)
                   - iqr: Interquartile range based
                   - mad: Median absolute deviation based
            threshold: Threshold for anomaly detection
                      - zscore: number of std deviations (default: 3.0)
                      - iqr: IQR multiplier (default: 1.5)
                      - mad: MAD multiplier (default: 3.0)
            ignore_nulls: Whether to ignore null values

        Returns:
            AnomalyReport with detected anomalies

        Example:
            >>> report = diag.detect_anomalies("Avg_CO2_lowpass", method="zscore")
            >>> print(f"Found {report.anomaly_count} anomalies")
        """
        logger.info(f"Detecting anomalies in '{column}' using {method}...")

        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")

        data = self.df[column].copy()
        if ignore_nulls:
            data = data.dropna()

        if len(data) == 0:
            logger.warning(f"No data in column '{column}'")
            return AnomalyReport(
                column=column,
                anomaly_count=0,
                anomaly_indices=[],
                method=method,
                threshold=threshold,
                summary_stats={},
            )

        # Detect anomalies based on method
        if method == "zscore":
            z_scores = np.abs((data - data.mean()) / data.std())
            anomaly_mask = z_scores > threshold
        elif method == "iqr":
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            anomaly_mask = (data < lower) | (data > upper)
        elif method == "mad":
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z = (
                0.6745 * (data - median) / mad
                if mad != 0
                else pd.Series(0, index=data.index)
            )
            anomaly_mask = np.abs(modified_z) > threshold
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'zscore', 'iqr', or 'mad'"
            )

        anomaly_indices = data.index[anomaly_mask].tolist()
        anomaly_values = data[anomaly_mask]

        # Summary statistics
        if len(anomaly_values) > 0:
            summary_stats = {
                "count": len(anomaly_values),
                "mean": float(anomaly_values.mean()),
                "std": float(anomaly_values.std()),
                "min": float(anomaly_values.min()),
                "max": float(anomaly_values.max()),
            }
        else:
            summary_stats = {"count": 0}

        logger.info(
            f"Found {len(anomaly_indices)} anomalies "
            f"({len(anomaly_indices) / len(data) * 100:.1f}%)"
        )

        return AnomalyReport(
            column=column,
            anomaly_count=len(anomaly_indices),
            anomaly_indices=anomaly_indices,
            method=method,
            threshold=threshold,
            summary_stats=summary_stats,
        )

    def get_statistical_summary(
        self, columns: Optional[List[ColumnName]] = None
    ) -> pd.DataFrame:
        """Generate statistical summary for specified columns.

        Calculates comprehensive statistics including mean, median, std,
        min, max, percentiles, skewness, and kurtosis.

        Args:
            columns: List of columns to summarize. If None, uses all numeric columns

        Returns:
            DataFrame with statistical summary

        Example:
            >>> summary = diag.get_statistical_summary(["Avg_CO2_lowpass", "Avg_SO2"])
            >>> print(summary)
        """
        logger.info("Generating statistical summary...")

        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        summary_data = []
        for col in columns:
            if col not in self.df.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue

            data = self.df[col].dropna()
            if len(data) == 0:
                continue

            summary_data.append(
                {
                    "column": col,
                    "count": len(data),
                    "mean": data.mean(),
                    "std": data.std(),
                    "min": data.min(),
                    "25%": data.quantile(0.25),
                    "50%": data.quantile(0.50),
                    "75%": data.quantile(0.75),
                    "max": data.max(),
                    "skewness": data.skew(),
                    "kurtosis": data.kurtosis(),
                    "missing_pct": (len(self.df) - len(data)) / len(self.df) * 100,
                }
            )

        return pd.DataFrame(summary_data)

    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive data health report.

        Combines multiple diagnostic checks into a single report with
        overall health score and recommendations.

        Returns:
            Dictionary with health metrics and recommendations

        Example:
            >>> report = diag.generate_health_report()
            >>> print(f"Health score: {report['health_score']}/100")
            >>> for rec in report['recommendations']:
            ...     print(f"- {rec}")
        """
        logger.info("Generating comprehensive health report...")

        # Run all analyses
        completeness = self.analyze_completeness()
        gaps = self.detect_gaps()
        calibrations = self.analyze_calibrations()

        # Calculate health score (0-100)
        health_score = 0.0

        # Completeness (40 points)
        health_score += completeness.completeness_pct * 0.4

        # Gap score (30 points)
        if gaps.total_gaps == 0:
            gap_score = 30.0
        else:
            # Penalize based on gap count and duration
            max_gap_hours = gaps.longest_gap.total_seconds() / 3600
            gap_penalty = min(30.0, gaps.total_gaps + max_gap_hours / 24)
            gap_score = max(0.0, 30.0 - gap_penalty)
        health_score += gap_score

        # Calibration score (30 points)
        if calibrations.zero_count > 0:
            cal_score = 30.0
        else:
            cal_score = 0.0
        health_score += cal_score

        # Generate recommendations
        recommendations = []
        if completeness.completeness_pct < 80:
            recommendations.append(
                f"Data completeness is {completeness.completeness_pct:.1f}%. "
                "Investigate data transmission or sensor issues."
            )
        if gaps.total_gaps > 10:
            recommendations.append(
                f"Found {gaps.total_gaps} data gaps. "
                "Check logger connectivity and power supply."
            )
        if gaps.longest_gap.total_seconds() > 86400:  # 1 day
            recommendations.append(
                f"Longest gap is {gaps.longest_gap.days} days. "
                "Review maintenance logs for this period."
            )
        if calibrations.zero_count == 0:
            recommendations.append(
                "No zero calibrations detected. Verify calibration schedule."
            )
        if calibrations.calibration_frequency:
            days_between = calibrations.calibration_frequency.days
            if days_between > 7:
                recommendations.append(
                    f"Calibration frequency is {days_between} days. "
                    "Consider more frequent calibrations."
                )

        if not recommendations:
            recommendations.append("No issues detected. Data quality is good.")

        logger.info(f"Health score: {health_score:.1f}/100")

        return {
            "health_score": round(health_score, 1),
            "completeness": {
                "pct": completeness.completeness_pct,
                "records": completeness.total_records,
                "expected": completeness.expected_records,
            },
            "gaps": {
                "count": gaps.total_gaps,
                "longest_hours": gaps.longest_gap.total_seconds() / 3600,
            },
            "calibrations": {
                "zero_count": calibrations.zero_count,
                "span_count": calibrations.span_count,
                "frequency_days": (
                    calibrations.calibration_frequency.days
                    if calibrations.calibration_frequency
                    else None
                ),
            },
            "recommendations": recommendations,
        }

    # Private helper methods

    def _find_missing_periods(
        self, expected_freq: Optional[str]
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Find periods with missing data."""
        if not expected_freq:
            return []

        try:
            expected_index = pd.date_range(
                start=self.df.index.min(), end=self.df.index.max(), freq=expected_freq
            )
            missing_times = expected_index.difference(self.df.index)

            # Group consecutive missing times into periods
            if len(missing_times) == 0:
                return []

            periods = []
            period_start = missing_times[0]
            prev_time = missing_times[0]

            for time in missing_times[1:]:
                if time - prev_time > pd.Timedelta(expected_freq):
                    periods.append((period_start, prev_time))
                    period_start = time
                prev_time = time

            periods.append((period_start, prev_time))
            return periods
        except Exception as e:
            logger.warning(f"Could not find missing periods: {e}")
            return []

    def _find_periods(
        self, df: pd.DataFrame, column: str, value: Any
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Find continuous periods where column equals value."""
        mask = df[column] == value
        if not mask.any():
            return []

        # Find start and end of continuous periods
        changes = mask.astype(int).diff()
        starts = df.index[changes == 1].tolist()
        ends = df.index[changes == -1].tolist()

        # Handle edge cases
        if mask.iloc[0]:
            starts.insert(0, df.index[0])
        if mask.iloc[-1]:
            ends.append(df.index[-1])

        return list(zip(starts, ends))
