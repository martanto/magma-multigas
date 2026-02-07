"""Time series plotting for volcanic gas data."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..data.dataset import Dataset
from .config import PlotConfig


class TimeSeriesPlotter:
    """Create time series plots for volcanic gas measurements.

    Provides high-level methods for common plot types with sensible defaults
    while allowing full customization of column names and styling.

    Example:
        >>> plotter = TimeSeriesPlotter(dataset)
        >>> plotter.plot_co2_so2_h2s()
        >>> plotter.save("figures/gas_plot.png")
    """

    def __init__(self, dataset: Dataset, config: Optional[PlotConfig] = None):
        """Initialize time series plotter.

        Args:
            dataset: Dataset to plot
            config: Plot configuration (uses defaults if None)
        """
        self.dataset = dataset
        self.config = config or PlotConfig()
        self._fig = None
        self._axes = None

        # Apply seaborn style
        sns.set_style(self.config.style)
        sns.set_context(self.config.context, font_scale=self.config.font_scale)

    def plot_co2_so2_h2s(
        self,
        co2_col: str = "Avg_CO2_lowpass",
        so2_col: str = "Avg_SO2",
        h2s_col: str = "Avg_H2S",
        plot_as_individual: bool = False,
        y_left_min: Optional[float] = None,
        y_left_max: Optional[float] = None,
        y_right_min: Optional[float] = None,
        y_right_max: Optional[float] = None,
        title: Optional[str] = None,
    ) -> "TimeSeriesPlotter":
        """Plot CO2, SO2, and H2S time series.

        Args:
            co2_col: Column name for CO2 data
            so2_col: Column name for SO2 data
            h2s_col: Column name for H2S data
            plot_as_individual: If True, create separate subplots
            y_left_min: Minimum value for left y-axis (CO2)
            y_left_max: Maximum value for left y-axis (CO2)
            y_right_min: Minimum value for right y-axis (SO2, H2S)
            y_right_max: Maximum value for right y-axis (SO2, H2S)
            title: Plot title (auto-generated if None)

        Returns:
            Self for method chaining
        """
        df = self.dataset.df

        if plot_as_individual:
            # Create three separate subplots
            self._fig, self._axes = plt.subplots(
                3,
                1,
                figsize=(self.config.width, self.config.height * 3),
                dpi=self.config.dpi,
                sharex=True,
            )

            # Plot CO2
            ax1 = self._axes[0]
            color = self.config.get_column_color(co2_col)
            label = self.config.get_column_label(co2_col)
            ax1.plot(df.index, df[co2_col], color=color, label=label, linewidth=2)
            ax1.set_ylabel(label, fontsize=self.config.label_fontsize, color=color)
            ax1.tick_params(
                axis="y", labelcolor=color, labelsize=self.config.tick_fontsize
            )
            ax1.grid(True, alpha=0.3)
            if y_left_min is not None or y_left_max is not None:
                ax1.set_ylim(y_left_min, y_left_max)

            # Plot SO2
            ax2 = self._axes[1]
            color = self.config.get_column_color(so2_col)
            label = self.config.get_column_label(so2_col)
            ax2.plot(df.index, df[so2_col], color=color, label=label, linewidth=2)
            ax2.set_ylabel(label, fontsize=self.config.label_fontsize, color=color)
            ax2.tick_params(
                axis="y", labelcolor=color, labelsize=self.config.tick_fontsize
            )
            ax2.grid(True, alpha=0.3)
            if y_right_min is not None or y_right_max is not None:
                ax2.set_ylim(y_right_min, y_right_max)

            # Plot H2S
            ax3 = self._axes[2]
            color = self.config.get_column_color(h2s_col)
            label = self.config.get_column_label(h2s_col)
            ax3.plot(df.index, df[h2s_col], color=color, label=label, linewidth=2)
            ax3.set_ylabel(label, fontsize=self.config.label_fontsize, color=color)
            ax3.tick_params(
                axis="y", labelcolor=color, labelsize=self.config.tick_fontsize
            )
            ax3.grid(True, alpha=0.3)
            ax3.set_xlabel("Date", fontsize=self.config.label_fontsize)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

            if title is None:
                title = f"Gas Measurements - {self.dataset.metadata.station}"
            self._fig.suptitle(title, fontsize=self.config.title_fontsize, y=0.995)

        else:
            # Create dual-axis plot
            self._fig, ax1 = plt.subplots(
                figsize=self.config.figsize, dpi=self.config.dpi
            )
            ax2 = ax1.twinx()

            # Plot CO2 on left axis
            color = self.config.get_column_color(co2_col)
            label = self.config.get_column_label(co2_col)
            ax1.plot(df.index, df[co2_col], color=color, label=label, linewidth=2)
            ax1.set_ylabel(label, fontsize=self.config.label_fontsize, color=color)
            ax1.tick_params(
                axis="y", labelcolor=color, labelsize=self.config.tick_fontsize
            )
            if y_left_min is not None or y_left_max is not None:
                ax1.set_ylim(y_left_min, y_left_max)

            # Plot SO2 and H2S on right axis
            color_so2 = self.config.get_column_color(so2_col)
            color_h2s = self.config.get_column_color(h2s_col)
            label_so2 = self.config.get_column_label(so2_col)
            label_h2s = self.config.get_column_label(h2s_col)

            ax2.plot(
                df.index, df[so2_col], color=color_so2, label=label_so2, linewidth=2
            )
            ax2.plot(
                df.index, df[h2s_col], color=color_h2s, label=label_h2s, linewidth=2
            )
            ax2.set_ylabel("SO2 / H2S", fontsize=self.config.label_fontsize)
            ax2.tick_params(axis="y", labelsize=self.config.tick_fontsize)
            if y_right_min is not None or y_right_max is not None:
                ax2.set_ylim(y_right_min, y_right_max)

            # Format x-axis
            ax1.set_xlabel("Date", fontsize=self.config.label_fontsize)
            ax1.tick_params(axis="x", labelsize=self.config.tick_fontsize)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc="upper left",
                fontsize=self.config.legend_fontsize,
            )

            if title is None:
                title = f"Gas Measurements - {self.dataset.metadata.station}"
            self._fig.suptitle(title, fontsize=self.config.title_fontsize, y=0.98)

            self._axes = [ax1, ax2]

        plt.tight_layout()
        return self

    def plot_gas_ratios(
        self,
        ratios: Optional[List[str]] = None,
        space_between_plots: float = 0.3,
        plot_regression: bool = True,
        title: Optional[str] = None,
    ) -> "TimeSeriesPlotter":
        """Plot gas ratios as stacked subplots.

        Args:
            ratios: List of ratio column names (uses defaults if None)
            space_between_plots: Vertical space between subplots
            plot_regression: If True, add trend lines
            title: Overall title (auto-generated if None)

        Returns:
            Self for method chaining
        """
        if ratios is None:
            ratios = [
                "Avg_CO2_H2S_ratio",
                "Avg_H2O_CO2_ratio",
                "Avg_H2S_SO2_ratio",
                "Avg_CO2_S_tot_ratio",
            ]

        # Filter to available columns
        ratios = [r for r in ratios if r in self.dataset.columns]

        if not ratios:
            raise ValueError("No ratio columns found in dataset")

        # Create subplots
        n_plots = len(ratios)
        self._fig, self._axes = plt.subplots(
            n_plots,
            1,
            figsize=(self.config.width, self.config.height * n_plots),
            dpi=self.config.dpi,
            sharex=True,
        )

        if n_plots == 1:
            self._axes = [self._axes]

        df = self.dataset.df

        for i, ratio_col in enumerate(ratios):
            ax = self._axes[i]

            # Get styling
            color = self.config.get_column_color(ratio_col)
            label = self.config.get_column_label(ratio_col)

            # Plot ratio
            ax.plot(
                df.index,
                df[ratio_col],
                color=color,
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=label,
            )

            # Add regression line if requested
            if plot_regression and df[ratio_col].notna().sum() > 1:
                # Remove NaN values
                valid_mask = df[ratio_col].notna()
                x_numeric = np.arange(len(df))[valid_mask]
                y_values = df[ratio_col][valid_mask].values

                if len(x_numeric) > 1:
                    # Calculate trend
                    z = np.polyfit(x_numeric, y_values, 1)
                    p = np.poly1d(z)
                    ax.plot(
                        df.index[valid_mask],
                        p(x_numeric),
                        "--",
                        color=color,
                        alpha=0.6,
                        linewidth=1,
                        label=f"Trend: y={z[0]:.4f}x+{z[1]:.2f}",
                    )

            # Styling
            ax.set_ylabel(label, fontsize=self.config.label_fontsize)
            ax.tick_params(axis="both", labelsize=self.config.tick_fontsize)
            ax.legend(fontsize=self.config.legend_fontsize, loc="best")
            ax.grid(True, alpha=0.3)

        # X-label on bottom plot only
        self._axes[-1].set_xlabel("Date", fontsize=self.config.label_fontsize)
        plt.setp(self._axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")

        if title is None:
            title = f"Gas Ratios - {self.dataset.metadata.station}"
        self._fig.suptitle(title, fontsize=self.config.title_fontsize, y=0.995)

        plt.tight_layout(h_pad=space_between_plots)
        return self

    def plot_columns(
        self,
        columns: List[str],
        separate: bool = True,
        colors: Optional[List[str]] = None,
        title: Optional[str] = None,
    ) -> "TimeSeriesPlotter":
        """Plot arbitrary columns as time series.

        Args:
            columns: List of column names to plot
            separate: If True, create separate subplots for each column
            colors: List of colors (auto-assigned if None)
            title: Plot title (auto-generated if None)

        Returns:
            Self for method chaining
        """
        df = self.dataset.df

        # Validate columns
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")

        if separate:
            # Create stacked subplots
            n_plots = len(columns)
            self._fig, self._axes = plt.subplots(
                n_plots,
                1,
                figsize=(self.config.width, self.config.height * n_plots),
                dpi=self.config.dpi,
                sharex=True,
            )

            if n_plots == 1:
                self._axes = [self._axes]

            for i, col in enumerate(columns):
                ax = self._axes[i]
                color = (
                    self.config.get_column_color(col) if colors is None else colors[i]
                )
                label = self.config.get_column_label(col)

                ax.plot(df.index, df[col], color=color, linewidth=2, label=label)
                ax.set_ylabel(label, fontsize=self.config.label_fontsize)
                ax.tick_params(axis="both", labelsize=self.config.tick_fontsize)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=self.config.legend_fontsize, loc="best")

            self._axes[-1].set_xlabel("Date", fontsize=self.config.label_fontsize)
            plt.setp(
                self._axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right"
            )

        else:
            # Single plot with multiple lines
            self._fig, ax = plt.subplots(
                figsize=self.config.figsize, dpi=self.config.dpi
            )

            for i, col in enumerate(columns):
                color = (
                    self.config.get_column_color(col) if colors is None else colors[i]
                )
                label = self.config.get_column_label(col)
                ax.plot(df.index, df[col], color=color, linewidth=2, label=label)

            ax.set_xlabel("Date", fontsize=self.config.label_fontsize)
            ax.set_ylabel("Value", fontsize=self.config.label_fontsize)
            ax.tick_params(axis="both", labelsize=self.config.tick_fontsize)
            ax.legend(fontsize=self.config.legend_fontsize, loc="best")
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

            self._axes = [ax]

        if title is None:
            title = f"Time Series - {self.dataset.metadata.station}"
        self._fig.suptitle(title, fontsize=self.config.title_fontsize, y=0.995)

        plt.tight_layout()
        return self

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
            self._axes = None
