"""Data availability and completeness visualization."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..data.dataset import Dataset
from .config import PlotConfig


class AvailabilityPlotter:
    """Visualize data availability and completeness.

    Creates plots showing:
    - Calendar heatmap of daily data counts
    - Time series of data availability
    - Completeness statistics per column
    - Missing data patterns

    Example:
        >>> plotter = AvailabilityPlotter(dataset)
        >>> plotter.plot_calendar_heatmap()
        >>> plotter.plot_completeness_bar()
        >>> plotter.save("figures/availability.png")
    """

    def __init__(self, dataset: Dataset, config: Optional[PlotConfig] = None, ax=None):
        """Initialize availability plotter.

        Args:
            dataset: Dataset to analyze
            config: Plot configuration (uses defaults if None)
            ax: Matplotlib axis (creates new figure if None)
        """
        self.dataset = dataset
        self.config = config or PlotConfig()
        self._ax = ax
        self._fig = None

        # Apply seaborn style
        sns.set_style(self.config.style)
        sns.set_context(self.config.context, font_scale=self.config.font_scale)

    def plot_calendar_heatmap(
        self,
        freq: str = "D",
        agg: str = "count",
        cmap: str = "YlGnBu",
        title: Optional[str] = None,
    ) -> "AvailabilityPlotter":
        """Plot calendar heatmap showing data availability.

        Args:
            freq: Frequency for aggregation ('D' for daily, 'W' for weekly)
            agg: Aggregation method ('count', 'sum', 'mean')
            cmap: Colormap name
            title: Plot title (auto-generated if None)

        Returns:
            Self for method chaining
        """
        # Resample data by frequency
        df = self.dataset.df
        resampled = df.resample(freq).size()

        # Create figure if needed
        if self._ax is None:
            self._fig, self._ax = plt.subplots(
                figsize=self.config.figsize, dpi=self.config.dpi
            )

        # Create date components for heatmap
        dates = resampled.index
        values = resampled.values

        # Group by month and day
        df_heatmap = pd.DataFrame(
            {"date": dates, "count": values, "year": dates.year, "month": dates.month}
        )

        # Plot heatmap
        pivot = df_heatmap.pivot_table(
            values="count", index="month", columns="year", aggfunc="sum"
        )

        sns.heatmap(
            pivot,
            cmap=cmap,
            annot=True,
            fmt=".0f",
            linewidths=0.5,
            cbar_kws={"label": "Record Count"},
            ax=self._ax,
        )

        # Set labels
        self._ax.set_xlabel("Year", fontsize=self.config.label_fontsize)
        self._ax.set_ylabel("Month", fontsize=self.config.label_fontsize)

        if title is None:
            title = f"Data Availability - {self.dataset.metadata.station}"
        self._ax.set_title(title, fontsize=self.config.title_fontsize, pad=20)

        plt.tight_layout()
        return self

    def plot_daily_counts(
        self,
        figsize: Optional[tuple] = None,
        color: str = "#039BE5",
        title: Optional[str] = None,
    ) -> "AvailabilityPlotter":
        """Plot time series of daily record counts.

        Args:
            figsize: Figure size (uses config default if None)
            color: Line color
            title: Plot title (auto-generated if None)

        Returns:
            Self for method chaining
        """
        # Calculate daily counts
        daily_counts = self.dataset.df.resample("D").size()

        # Create figure if needed
        if self._ax is None:
            figsize = figsize or self.config.figsize
            self._fig, self._ax = plt.subplots(figsize=figsize, dpi=self.config.dpi)

        # Plot line
        self._ax.plot(daily_counts.index, daily_counts.values, color=color, linewidth=2)

        # Fill area under curve
        self._ax.fill_between(
            daily_counts.index, daily_counts.values, alpha=0.3, color=color
        )

        # Set labels
        self._ax.set_xlabel("Date", fontsize=self.config.label_fontsize)
        self._ax.set_ylabel("Record Count", fontsize=self.config.label_fontsize)

        if title is None:
            title = f"Daily Data Counts - {self.dataset.metadata.station}"
        self._ax.set_title(title, fontsize=self.config.title_fontsize, pad=20)

        # Format x-axis
        self._ax.tick_params(axis="both", labelsize=self.config.tick_fontsize)
        plt.setp(self._ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        return self

    def plot_completeness_bar(
        self,
        columns: Optional[list] = None,
        threshold: float = 0.0,
        figsize: Optional[tuple] = None,
        title: Optional[str] = None,
    ) -> "AvailabilityPlotter":
        """Plot bar chart of data completeness per column.

        Args:
            columns: Columns to analyze (all numeric columns if None)
            threshold: Minimum completeness to include (0-1)
            figsize: Figure size (uses config default if None)
            title: Plot title (auto-generated if None)

        Returns:
            Self for method chaining
        """
        # Select columns
        if columns is None:
            columns = self.dataset.df.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        # Calculate completeness
        total_rows = len(self.dataset)
        completeness = {}

        for col in columns:
            non_null = self.dataset.df[col].notna().sum()
            completeness[col] = (non_null / total_rows) * 100

        # Filter by threshold
        completeness = {k: v for k, v in completeness.items() if v >= threshold * 100}

        if not completeness:
            raise ValueError(f"No columns meet threshold of {threshold * 100}%")

        # Sort by completeness
        completeness = dict(sorted(completeness.items(), key=lambda x: x[1]))

        # Create figure if needed
        if self._ax is None:
            figsize = figsize or (self.config.width, len(completeness) * 0.4 + 2)
            self._fig, self._ax = plt.subplots(figsize=figsize, dpi=self.config.dpi)

        # Create color map based on completeness
        colors = plt.cm.RdYlGn(np.array(list(completeness.values())) / 100)

        # Plot horizontal bar
        bars = self._ax.barh(list(completeness.keys()), list(completeness.values()))

        # Color bars
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # Add percentage labels
        for i, (col, pct) in enumerate(completeness.items()):
            self._ax.text(
                pct + 1,
                i,
                f"{pct:.1f}%",
                va="center",
                fontsize=self.config.tick_fontsize,
            )

        # Set labels
        self._ax.set_xlabel("Completeness (%)", fontsize=self.config.label_fontsize)
        self._ax.set_xlim(0, 105)

        if title is None:
            title = f"Data Completeness - {self.dataset.metadata.station}"
        self._ax.set_title(title, fontsize=self.config.title_fontsize, pad=20)

        self._ax.tick_params(axis="both", labelsize=self.config.tick_fontsize)
        plt.tight_layout()
        return self

    def plot_missing_patterns(
        self,
        columns: Optional[list] = None,
        max_columns: int = 20,
        figsize: Optional[tuple] = None,
        title: Optional[str] = None,
    ) -> "AvailabilityPlotter":
        """Plot missing data patterns as heatmap.

        Args:
            columns: Columns to analyze (auto-selected if None)
            max_columns: Maximum columns to display
            figsize: Figure size (uses config default if None)
            title: Plot title (auto-generated if None)

        Returns:
            Self for method chaining
        """
        # Select columns with missing data
        if columns is None:
            missing_counts = self.dataset.df.isnull().sum()
            columns = missing_counts[missing_counts > 0].index.tolist()[:max_columns]

        if not columns:
            raise ValueError("No missing data found in dataset")

        # Create missing data matrix (subsample if too large)
        df = self.dataset.df[columns]
        if len(df) > 1000:
            # Sample evenly across the dataset
            idx = np.linspace(0, len(df) - 1, 1000, dtype=int)
            df = df.iloc[idx]

        missing_matrix = df.isnull().astype(int)

        # Create figure if needed
        if self._ax is None:
            figsize = figsize or (self.config.width, 8)
            self._fig, self._ax = plt.subplots(figsize=figsize, dpi=self.config.dpi)

        # Plot heatmap
        sns.heatmap(
            missing_matrix.T,
            cmap="RdYlGn_r",
            cbar_kws={"label": "Missing (1) / Present (0)"},
            yticklabels=columns,
            xticklabels=False,
            ax=self._ax,
        )

        # Set labels
        self._ax.set_xlabel("Record Index", fontsize=self.config.label_fontsize)
        self._ax.set_ylabel("Column", fontsize=self.config.label_fontsize)

        if title is None:
            title = f"Missing Data Patterns - {self.dataset.metadata.station}"
        self._ax.set_title(title, fontsize=self.config.title_fontsize, pad=20)

        plt.tight_layout()
        return self

    def get_statistics(self) -> pd.DataFrame:
        """Get availability statistics for all columns.

        Returns:
            DataFrame with completeness statistics
        """
        total_rows = len(self.dataset)
        stats = []

        for col in self.dataset.columns:
            non_null = self.dataset.df[col].notna().sum()
            null_count = total_rows - non_null
            completeness_pct = (non_null / total_rows) * 100

            stats.append(
                {
                    "column": col,
                    "total_records": total_rows,
                    "available": non_null,
                    "missing": null_count,
                    "completeness_pct": completeness_pct,
                }
            )

        return pd.DataFrame(stats).sort_values("completeness_pct", ascending=False)

    def save(self, path: str, bbox_inches: str = "tight", **kwargs) -> Path:
        """Save the current plot.

        Args:
            path: Output file path
            bbox_inches: Bounding box ('tight' removes whitespace)
            **kwargs: Additional arguments for savefig

        Returns:
            Path to saved file
        """
        if self._fig is None:
            raise ValueError("No plot to save. Create a plot first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._fig.savefig(path, bbox_inches=bbox_inches, dpi=self.config.dpi, **kwargs)
        plt.close(self._fig)

        return path

    def show(self):
        """Display the current plot."""
        if self._fig is None:
            raise ValueError("No plot to show. Create a plot first.")
        plt.show()

    def close(self):
        """Close the current figure."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
