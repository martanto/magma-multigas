import numpy as np
import pandas as pd
import os

from .variables import plot_properties
from .resources.colors import generate_random_color
from matplotlib import pyplot as plt
from matplotlib.dates import date2num


class Plot:
    def __init__(self, df: pd.DataFrame = None, y_min: float = None,
                 y_max: float = None, y_max_multiplier: float = 1,
                 width: int = 12, height: int = 4):
        self.df = df
        self.y_min = y_min
        self.y_max = y_max
        self.y_max_multiplier = y_max_multiplier
        self.width = width
        self.height = height

        self.start_date = df.index[0]
        self.end_date = df.index[-1]

        self.start_date_str = self.start_date.strftime('%Y-%m-%d')
        self.end_date_str = self.end_date.strftime('%Y-%m-%d')

        figures_dir = os.path.join(os.getcwd(), 'figures')
        os.makedirs(figures_dir, exist_ok=True)

        self.figures_dir = figures_dir

    def ax_scatter(self, ax: plt.Axes, column: str) -> plt.Axes:
        """Scatter plot.

        Args:
            ax (plt.Axes): Axes object.
            column (str): Column name.

        Returns:
            plt.Axes: Axes object.
        """
        kwargs = {
            'c': generate_random_color(),
            'alpha': 0.5,
            'label': column,
            's': 20
        }

        if column in plot_properties.keys():
            kwargs = {
                'c': plot_properties[column]['color'],
                'alpha': 0.5,
                'label': plot_properties[column]['label'],
                's': 20
            }

        df = self.df
        y_min = self.y_min
        y_max = self.y_max

        y_min = df[column].min() if y_min is None else y_min
        y_max = df[column].max() * self.y_max_multiplier if y_max is None else y_max

        ax.scatter(df.index, df[column], **kwargs)
        ax.set_xlim(date2num(self.start_date), date2num(self.end_date))

        if df[column].sum() > 0:
            ax.set_ylim(y_min, y_max)

        ax.legend(loc=2, fontsize=8)
        ax.grid(True, which='both', linestyle='-', linewidth=1, alpha=1)
        ax.yaxis.get_major_ticks()[0].label1.set_visible(False)

        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=30, horizontalalignment='right')

        return ax

    def ax_plot(self, ax: plt.Axes, column: str) -> plt.Axes:
        """Line plot.

        Args:
            ax (plt.Axes): Axes object.
            column (str): Column name.

        Returns:
            plt.Axes: Axes object.
        t"""
        kwargs = {
            'color': generate_random_color(),
            'marker': 'D',
            'label': column,
            'linestyle': '-.'
        }

        if column in plot_properties.keys():
            kwargs = {
                'color': plot_properties[column]['color'],
                'marker': plot_properties[column]['marker'],
                'label': plot_properties[column]['label'],
                'linestyle': '-.'
            }

        df = self.df
        y_min = self.y_min
        y_max = self.y_max

        y_min = df[column].min() if y_min is None else y_min
        y_max = df[column].max() * self.y_max_multiplier if y_max is None else y_max

        ax.plot(df.index, df[column], **kwargs)
        ax.set_xlim(date2num(self.start_date), date2num(self.end_date))

        if df[column].sum() > 0:
            ax.set_ylim(y_min, y_max)

        ax.legend(loc=2, fontsize=8)
        ax.grid(True, which='both', linestyle='-', linewidth=1, alpha=1)
        ax.yaxis.get_major_ticks()[0].label1.set_visible(False)

        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=30, horizontalalignment='right')

        return ax

    def ax_co2_so2_h2s(self, ax: plt.Axes) -> plt.Axes:
        """Specific plot for Average value of CO2, SO2, and H2S.

        Args:
            ax (plt.Axes):

        Returns:
            plt.Axes:
        """
        df = self.df
        y_min = self.y_min
        y_max = self.y_max

        y_min = df['Avg_CO2_lowpass'].min() if y_min is None else y_min
        y_max = df['Avg_CO2_lowpass'].max() * self.y_max_multiplier if y_max is None else y_max

        ax.plot(df.index, df['Avg_CO2_lowpass'], color='#039BE5',
                linestyle='--', marker='D', label='Average CO2 Lowpass')
        ax.set_xlim(date2num(self.start_date), date2num(self.end_date))
        ax.set_ylim(y_min, y_max)
        ax.legend(loc=2, fontsize=8)

        ax_right = ax.twinx()
        ax_right.plot(df.index, df['Avg_H2S'], color='#F44336', marker='^', label='Average H2S')
        ax_right.plot(df.index, df['Avg_SO2'], color='#8BC34A', marker='*', label='Average SO2', )
        ax_right.legend(loc=1, fontsize=8)

        ax.grid(True, which='both', linestyle='--', alpha=1)
        ax.set_title("6 Hours Average\n $CO_{2}$ - $H_{2}S$ - $SO_{2}$ Concentration (ppm)")
        return ax

    def plot_co2_so2_h2s(self, plot_as_individual: bool = False, space_between_plot: float = None) -> str:
        """Plot Average CO2, SO2, and H2S using six hours of data.

        Args:
            plot_as_individual (bool):
            space_between_plot (float):

        Returns:
            Figure save location
        """
        if plot_as_individual is True:
            columns = ['Avg_CO2_lowpass', 'Avg_H2S', 'Avg_SO2']
            figsize = (self.width, self.height * len(columns))
            fig, axs = plt.subplots(nrows=len(columns), ncols=1, figsize=figsize, sharex=True)
            fig.suptitle("6 Hours Average\n $CO_{2}$ - $H_{2}S$ - "
                         "$SO_{2}$ Concentration (ppm)")
            for index, column in enumerate(columns):
                self.ax_plot(ax=axs[index], column=column)

            if space_between_plot is not None:
                plt.subplots_adjust(hspace=space_between_plot)

        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(self.width, self.height))
            self.ax_co2_so2_h2s(ax)

        filename = '{}_{}_co2_so2_h2s_concentration'.format(self.start_date_str, self.end_date_str)

        return self.save(fig, filename)

    def plot_gas_ratio(self) -> str:
        """Plot ratio average value of CO2/H2S, H2O/CO2, H2S/SO2,
        CO2/SO2 and CO2/S-Total Concentration (ppm)

        Returns:
            Figure save location
        """
        columns = ['Avg_CO2_H2S_ratio', 'Avg_H2O_CO2_ratio', 'Avg_H2S_SO2_ratio',
                   'Avg_CO2_SO2_ratio', 'Avg_CO2_S_tot_ratio']

        return self.columns(columns=columns)

    def plot_columns(self, columns: str | list[str], plot_type: str = 'scatter') -> str:
        """Plot for selected columns

        Args:
            columns (str | list[str]): Columns to plot
            plot_type (str): Plot type. Choose between 'scatter' or 'plot'.

        Returns:
            Figure save location
        """
        figsize = (self.width, self.height * len(columns))
        fig, axs = plt.subplots(nrows=len(columns), ncols=1, figsize=figsize, sharex=True)

        filename = '{}_{}_{}'.format(self.start_date_str, self.end_date_str, '_'.join(columns))

        if len(columns) == 1:
            self.ax_scatter(ax=axs, column=columns[0]) if plot_type == 'scatter' \
                else self.ax_plot(ax=axs, column=columns[0])
        else:
            for index, column in enumerate(columns):
                self.ax_scatter(ax=axs[index], column=column) if plot_type == 'scatter' \
                    else self.ax_plot(ax=axs[index], column=column)

        return self.save(fig, filename)

    def save(self, fig: plt.Figure, filename: str) -> str:
        """Save figure to file.

        Args:
            fig (plt.Figure): Figure to save
            filename (str): Filename to save

        Returns:
            Figure save location
        """
        save_path = os.path.join(self.figures_dir, f"{filename}.png")
        fig.savefig(save_path, dpi=300)

        return save_path
