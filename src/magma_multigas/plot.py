import pandas as pd
import os
import seaborn as sns

from .variables import plot_properties
from .resources.colors import generate_random_color
from .utilities import y_prediction, regression_function, all_evaluations
from .validator import validate_column_name
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
from datetime import datetime
from typing import List, Self

sns.set_style("whitegrid")


def default_filename(prefix: str = None, suffix: str = None) -> str:
    """Generate a unique filename.

    Args:
        prefix (str, optional): A prefix to prepend to the filename. Defaults to None.
        suffix (str, optional): A suffix to append to the filename. Defaults to None.

    Returns:
        str: A unique filename.
    """
    prefix = 'plot' if suffix is None else suffix

    suffix = '' if suffix is None else suffix

    if not prefix.endswith('_'):
        prefix = prefix + '_'

    if (len(suffix) > 0) and (not suffix.startswith('_')):
        suffix = '_' + suffix

    current_datetime: str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{prefix}{current_datetime}{suffix}"
    filename = filename[:100]
    return filename


class Plot:
    def __init__(self, df: pd.DataFrame = None, width: int = 12, height: int = 4,
                 y_max_multiplier: float = 1.0, datetime_column: str = 'TIMESTAMP',
                 margins: float = 0.05):
        """Plot data from dataframe.

        Args:
            df (pd.DataFrame): A dataframe containing data to be plotted.
            width (int, optional): Width of the plot. Defaults to 12.
            height (int, optional): Height of the plot or column. Defaults to 4.
            y_max_multiplier (float, optional): Maximum y-axis multiplier. Defaults to 1.0.
            datetime_column (str, optional): A column name containing the datetime. Defaults to 'TIMESTAMP'.
            margins (float, optional): Margin of the plot. Defaults to 0.05.
        """
        self.df_original = df.copy()
        self.df = df

        self.width = width
        self.height = height

        self.datetime_column = datetime_column
        self.margins = margins

        self.x_limit: List[float] | None = None
        self.y_limit: List[float] | None = None

        self.start_date = df.index[0]
        self.end_date = df.index[-1]

        self.start_date_str = self.start_date.strftime('%Y-%m-%d')
        self.end_date_str = self.end_date.strftime('%Y-%m-%d')

        figures_dir = os.path.join(os.getcwd(), 'figures')
        os.makedirs(figures_dir, exist_ok=True)

        self.y_max_multiplier: float = y_max_multiplier
        self.filename: str = default_filename()
        self.figures_dir = figures_dir

    @property
    def data(self) -> pd.DataFrame:
        """Returns plot dataframe. Alias for df attribute."""
        return self.df

    def _ax_scatter(self, df: pd.DataFrame, ax: plt.Axes, column: str) -> plt.Axes:
        """Scatter plot.

        Args:
            ax (plt.Axes): Axes object.
            column (str): Column name.

        Returns:
            plt.Axes: Axes object.
        """
        marker_size = 60
        alpha = 0.5

        kwargs = {
            'c': generate_random_color(),
            'alpha': alpha,
            'label': column,
            's': marker_size
        }

        if column in plot_properties.keys():
            kwargs = {
                'c': plot_properties[column]['color'],
                'alpha': alpha,
                'label': plot_properties[column]['label'],
                's': marker_size
            }

        if not isinstance(df.index, pd.DatetimeIndex):
            kwargs['label'] = 'Ratio Value'
            _start_date = df[self.datetime_column].iloc[0].strftime('%d %b %Y, %H:%M:%S')
            _end_date = df[self.datetime_column].iloc[-1].strftime('%d %b %Y, %H:%M:%S')
            _datetime = f"{_start_date} - {_end_date} UTC"
            title = f"{column}/{df.index.name}\n{_datetime}"
            ax.set_title(title)

        ax.scatter(df.index, df[column], edgecolors="k", **kwargs)

        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=30, horizontalalignment='right')

        return ax

    def _ax_plot(self, df: pd.DataFrame, ax: plt.Axes, column: str) -> plt.Axes:
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

        ax.plot(df.index, df[column], **kwargs)

        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=30, horizontalalignment='right')

        return ax

    def _ax_co2_so2_h2s(self, df: pd.DataFrame, ax: plt.Axes,
                        y_left_min: float = None, y_left_max: float = None,
                        y_right_min: float = None, y_right_max: float = None) -> plt.Axes:
        """Specific plot for Average value of CO2, SO2, and H2S.

        Args:
            df (pd.DataFrame): Dataframe.
            ax (plt.Axes): Axe figures
            y_left_min (float): Minimum value for the left y-axis of the plot.
            y_left_max (float): Maximum value for the left y-axis of the plot.
            y_right_min (float): Minimum value for the right y-axis of the plot.
            y_right_max (float): Maximum value for the right y-axis of the plot.

        Returns:
            plt.Axes:
        """
        y_left_min = df['Avg_CO2_lowpass'].min() if y_left_min is None else y_left_min
        y_left_max = df['Avg_CO2_lowpass'].max() * self.y_max_multiplier if y_left_max is None else y_left_max

        # Left axes
        ax_left = ax
        ax_left.plot(df.index, df['Avg_CO2_lowpass'],
                color=plot_properties['Avg_CO2_lowpass']['color'],
                marker=plot_properties['Avg_CO2_lowpass']['marker'],
                label=plot_properties['Avg_CO2_lowpass']['label'],
                linestyle='--')

        ax_left.legend(loc=2, fontsize=8)
        ax_left.set_xlim(date2num(self.start_date), date2num(self.end_date))
        ax_left.set_ylim(y_left_min, y_left_max)

        # Right Axes
        ax_right = ax.twinx()
        ax_right.plot(df.index, df['Avg_H2S'],
                      color=plot_properties['Avg_H2S']['color'],
                      marker=plot_properties['Avg_H2S']['marker'],
                      label=plot_properties['Avg_H2S']['label'])

        ax_right.plot(df.index, df['Avg_SO2'],
                      color=plot_properties['Avg_SO2']['color'],
                      marker=plot_properties['Avg_SO2']['marker'],
                      label=plot_properties['Avg_SO2']['label'])

        minimum_h2s = df['Avg_H2S'].min()
        maximum_h2s = df['Avg_H2S'].max()
        minimum_so2 = df['Avg_SO2'].min()
        maximum_so2 = df['Avg_SO2'].max()

        minimum = minimum_h2s if (minimum_h2s < minimum_so2) else minimum_so2
        maximum = maximum_h2s if (maximum_h2s > maximum_so2) else maximum_so2

        y_right_min = minimum if y_right_min is None else y_right_min
        y_right_max = maximum * self.y_max_multiplier if y_right_max is None else y_right_max
        ax_right.set_ylim(y_right_min, y_right_max)

        ax_right.legend(loc=1, fontsize=8)
        return ax

    def plot_co2_so2_h2s(self, y_left_min: float = None, y_left_max: float = None,
                         y_right_min: float = None, y_right_max: float = None,
                         plot_as_individual: bool = False,
                         space_between_plot: float = None,
                         filename: str = None,
                         y_max_multiplier: float = None) -> Self:
        """Plot Average CO2, SO2, and H2S using six hours of data.

        Args:
            y_left_min (float): Minimum value for the left y-axis of the plot.
            y_left_max (float): Maximum value for the left y-axis of the plot.
            y_right_min (float): Minimum value for the right y-axis of the plot.
            y_right_max (float): Maximum value for the right y-axis of the plot.
            plot_as_individual (bool): plot would be saved individually
            space_between_plot (float): space between plot
            filename (str): filename to save the figure.
            y_max_multiplier (float): Max multiplier. Default is 1.0

        Returns:
            Plot class
        """
        df = self.df.copy()

        if filename is None:
            filename = ('{}_{}_co2_so2_h2s_concentration'
                        .format(self.start_date_str, self.end_date_str))

        if y_max_multiplier is not None:
            self.y_max_multiplier = y_max_multiplier

        if plot_as_individual is True:
            columns = ['Avg_CO2_lowpass', 'Avg_H2S', 'Avg_SO2']

            title = (f"6 Hours Average\n $CO_{2}$ - $H_{2}S$ - "
                     f"$SO_{2}$ Concentration (ppm)")

            self.plot_columns(df=df,
                              columns=columns,
                              plot_type='plot',
                              filename=filename,
                              title=title,
                              space_between_plot=space_between_plot,
                              y_max_multiplier=y_max_multiplier,
                              plot_regression=False)

            # figsize = (self.width, self.height * len(columns))
            # fig, axs = plt.subplots(nrows=len(columns), ncols=1,
            #                         figsize=figsize, sharex=True)
            # fig.suptitle("6 Hours Average\n $CO_{2}$ - $H_{2}S$ - "
            #              "$SO_{2}$ Concentration (ppm)")
            #
            # for index, column in enumerate(columns):
            #     ax = self._ax_plot(df=df, ax=axs[index], column=column)
            #     if df[column].sum() > 0:
            #         ax = self._set_y_limit(ax=ax, df=df, column=column, y_min=y_left_min, y_max=y_left_max)
            #
            # if space_between_plot is not None:
            #     plt.subplots_adjust(hspace=space_between_plot)

        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(self.width, self.height))
            self._ax_co2_so2_h2s(df=df, ax=ax, y_left_min=y_left_min, y_left_max=y_left_max,
                                 y_right_min=y_right_min, y_right_max=y_right_max)

            ax.grid(True, which='both', linestyle='-.', alpha=1)
            fig.suptitle("6 Hours Average\n $CO_{2}$ - $H_{2}S$ - "
                         "$SO_{2}$ Concentration (ppm)")

        self.df = df

        print(f"✅ Figure saved to: {filename} ")

        return self

    def plot_gas_ratio(self, plot_type: str = 'scatter',
                       filename: str = None,
                       space_between_plot: float = None,
                       y_max_multiplier: float = None,
                       plot_regression: bool = False) -> Self:
        """Plot ratio average value of CO2/H2S, H2O/CO2, H2S/SO2,
        CO2/SO2 and CO2/S-Total Concentration (ppm)

        Args:
            plot_type (str): Plot type. Choose between 'scatter' or 'plot'.
            filename (str, optional): filename to save the figure.
            space_between_plot (float, optional): space between plot
            y_max_multiplier (float, optional): Max multiplier. Default is 1.0
            plot_regression (bool, optional): Plot regression.

        Returns:
            Plot class
        """
        df = self.df.copy()

        columns = ['Avg_CO2_H2S_ratio', 'Avg_H2O_CO2_ratio', 'Avg_H2S_SO2_ratio',
                   'Avg_CO2_SO2_ratio', 'Avg_CO2_S_tot_ratio']

        if filename is None:
            filename = '{}_{}_{}'.format(self.start_date_str,
                                         self.end_date_str, '_'.join(columns))

        self.plot_columns(df=df,
                          columns=columns, plot_type=plot_type,
                          filename=filename,
                          space_between_plot=space_between_plot,
                          y_max_multiplier=y_max_multiplier,
                          plot_regression=plot_regression)

        return self

    def _set_x_limit(self, ax: plt.Axes, df: pd.DataFrame,
                     x_min: float = None, x_max: float = None) -> plt.Axes:

        if (x_min is None) and (x_max is None):
            return ax

        x_min = x_min if x_min is not None else df.idxmin()
        x_max = x_max if x_max is not None else df.idxmax()

        self.x_limit = [x_min, x_max]

        ax.set_xlim(x_min, x_max)

        return ax

    def _set_y_limit(self, ax: plt.Axes, df: pd.DataFrame, column: str,
                     y_min: float = None, y_max: float = None) -> plt.Axes:

        if (y_min is None) and (y_max is None):
            return ax

        y_max_multiplier = self.y_max_multiplier

        y_min = y_min if y_min is not None else df[column].min()
        y_max = y_max if y_max is not None else df[column].max() * y_max_multiplier

        self.y_limit = [y_min, y_max]

        ax.set_ylim(y_min, y_max)

        return ax

    def plot_columns(self, columns: str | list[str],
                     df: pd.DataFrame = None,
                     x_min: float = None,
                     x_max: float = None,
                     y_min: float = None,
                     y_max: float = None,
                     y_max_multiplier: float = None,
                     plot_type: str = 'scatter',
                     title: str = None,
                     filename: str = None,
                     space_between_plot=None,
                     plot_regression: bool = False,
                     disable_regression_for: List[str] = None,
                     validate_column: bool = True) -> pd.DataFrame:
        """Plot for selected columns

        Args:
            df (pd.DataFrame): DataFrame to plot
            columns (str | list[str]): Columns to plot
            plot_type (str): Plot type. Choose between 'scatter' or 'plot'.
            title (str, optional): Plot title.
            filename (str): filename to save the figure.
            space_between_plot (float, optional): space between plot
            x_min (float, optional): Minimum x value. Defaults to None.
            x_max (float, optional): Maximum x value. Defaults to None.
            y_min (float, optional): Min. Minimum value. Defaults to None.
            y_max (float, optional): Max. Maximum value. Defaults to None.
            y_max_multiplier (float): Max multiplier. Default is 1.0
            plot_regression (bool): Plot regression
            disable_regression_for (list[str]): List of columns disable regression
            validate_column (bbol): Validating column is exists or not

        Returns:
            Plot dataframe
        """
        if df is None:
            df = self.df.copy()

        df = df.sort_index().drop_duplicates(keep='last')

        if disable_regression_for is None:
            disable_regression_for = []

        if validate_column is True:
            for column in columns:
                validate_column_name(column_name=column, column_list=df.columns.tolist())
                if column == self.datetime_column:
                    columns.remove(column)

                empty = True if (df[column].sum() == 0) else False
                if empty:
                    print(f"⚠️ Column ['{column}'] is empty. "
                          f"Column '{column}' will not be plotted.")
                    columns.remove(column)

        if isinstance(columns, str):
            columns = [columns]

        if y_max_multiplier is not None:
            self.y_max_multiplier = y_max_multiplier

        remove_gap: bool = True if space_between_plot == 0 else False

        figsize = (self.width, self.height * len(columns))
        fig, axs = plt.subplots(nrows=len(columns), ncols=1,
                                figsize=figsize, sharex=True)

        if title is not None:
            fig.suptitle(title)

        for index, column in enumerate(columns):
            ax = axs if (len(columns) == 1) else axs[index]

            ax = self._ax_scatter(df=df, ax=ax, column=column) if plot_type == 'scatter' \
                else self._ax_plot(df=df, ax=ax, column=column)

            # Plot Regression
            if (plot_regression is True) and (column not in disable_regression_for):
                ax.plot(df.index, y_prediction(df, column),
                        label=regression_function(df, column))

                errors = all_evaluations(df, column)
                rmse = errors['rmse']
                r2 = errors['r2']

                ax.annotate(
                    text=f'RMSE = {rmse:.3f}, $R^2$ = {r2:.3f}',
                    xy=(0.02, 0.8),
                    xycoords='axes fraction',
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.5)
                )

                # Removing 0 values if remove_gap is True
                if remove_gap is True:
                    ax.yaxis.get_major_ticks()[0].label1.set_visible(False)

            if isinstance(df.index, pd.DatetimeIndex):
                ax.set_xlim(date2num(self.start_date), date2num(self.end_date))
            else:
                ax.set_xlabel(df.index.name)
                ax.set_ylabel(column)
                ax = self._set_x_limit(ax=ax, df=df, x_min=x_min, x_max=x_max)

            if df[column].sum() > 0:
                ax = self._set_y_limit(ax=ax, df=df, column=column, y_min=y_min, y_max=y_max)

            ax.margins(self.margins)
            ax.legend(loc='upper right', fontsize=10, ncol=4)
            ax.grid(True, which='both', linestyle='-.', alpha=1)

        if remove_gap is False:
            plt.subplots_adjust(hspace=space_between_plot)

        if filename is None:
            filename = default_filename(prefix='plot_columns')

        print(f"✅ Figure saved to: {filename} ")

        self.df = df

        return df

    def plot_between_two_columns(self, x: str, y: str,
                                 x_min: float = None, x_max: float = None,
                                 y_min: float = None, y_max: float = None,
                                 df: pd.DataFrame = None,
                                 filename: str = None,
                                 plot_regression: bool = False) -> Self:
        """Plot between two selected columns.

        Args:
            df (pd.DataFrame): DataFrame to plot
            x (str): Column name
            y (str): Column name
            x_min (float, optional): Minimum x-value. Defaults to None.
            x_max (float, optional): Maximum x-value. Defaults to None.
            y_min (float, optional): Minimum y-value. Defaults to None.
            y_max (float, optional): Maximum y-value. Defaults to None.
            filename (str): filename to save the figure.
            plot_regression (bool): Plot regression

        Returns:
            Plot class
        """
        if df is None:
            df = self.df.copy()

        columns_list = df.columns.tolist()

        for column in [x, y]:
            validate_column_name(column_name=column, column_list=columns_list)
            empty = True if (df[column].sum() == 0) else False
            if empty:
                raise ValueError(f"⛔ Column ['{column}'] is empty. "
                                 f"Please check colum value and change column {column}.")

        df = df[[x, y]]
        df = df[(df[x] > 0) & (df[y] > 0)]
        df = df.reset_index().set_index(x)

        self.plot_columns(df=df, columns=y, x_min=x_min, x_max=x_min,
                          y_min=y_min, y_max=y_max, filename=filename,
                          plot_regression=plot_regression, validate_column=False)

        return self

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
