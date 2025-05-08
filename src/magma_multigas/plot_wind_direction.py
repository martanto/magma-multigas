import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib import colormaps
from magma_multigas import MultiGasData


class PlotWindDirection:
    def __init__(self, csv: str, start_date: str | None = None, end_date: str | None = None,
                 type_of_data: str = 'six_hours', wind_direction_column: str = 'Avg_Wind_Direction',
                 wind_speed_column: str = 'Avg_Wind_Speed', column_values: str | list[str] = 'Avg_CO2_S_tot_ratio',
                 volcano_name: str | None = None, current_dir: str | None = None):
        multigas_data = MultiGasData(type_of_data, csv)

        if isinstance(column_values, str):
            column_values = column_values.split(',')
        selected_column_names: list[str] = [wind_direction_column, wind_speed_column] + column_values

        self.wind_speed_column = wind_speed_column
        self.wind_direction_column = wind_direction_column
        self.column_names: list[str] = column_values
        self.df_original = multigas_data.df.copy()
        self.start_date = self.df_original.index[0]
        self.end_date = self.df_original.index[-1]

        if start_date is None and end_date is None:
            self.df = self.df_original
        else:
            self.df = (multigas_data.where_date_between(start_date=start_date, end_date=end_date)
                              .select_columns(column_names=selected_column_names)).df
            self.start_date: pd.Timestamp = self.df.index[0]
            self.start_date: pd.Timestamp = self.df.index[-1]

        self.volcano_name: str | None = volcano_name
        self.current_dir: str | None = os.getcwd() if current_dir is None else current_dir
        self.output_dir, self.figures_dir = self.check_directory()

    def check_directory(self) -> tuple[str, str]:

        current_dir = self.current_dir

        output_dir = os.path.join(current_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        figures_dir = os.path.join(output_dir, 'figures', 'wind_direction')
        os.makedirs(figures_dir, exist_ok=True)

        return output_dir, figures_dir

    @property
    def wind_speed(self) -> pd.DataFrame:
        return self.df[self.wind_speed_column]

    @property
    def wind_direction(self) -> pd.DataFrame:
        return self.df[self.wind_direction_column]

    @property
    def u_vector(self):
        return -np.sin(np.radians(self.wind_direction))

    @property
    def v_vector(self):
        return -np.cos(np.radians(self.wind_direction))

    def show(self, color: str = 'viridis', width: int = 12, height: int = 3,
             save_plot: bool = True, interval: int = 1,
             hspace: float = 0.0, prefix: str = 'wind_direction',
             dpi: int = 300, filename: str | None = None) -> tuple[plt.Figure, plt.Axes]:

        df = self.df
        wind_speed = self.wind_speed
        u = self.u_vector
        v = self.v_vector
        total_columns: int = len(self.column_names)

        # Create a colormap based on wind speed
        norm = Normalize(vmin=np.min(wind_speed), vmax=np.max(wind_speed))
        cmap = colormaps[color]

        fig, axs = plt.subplots(
            nrows=total_columns,
            ncols=1,
            figsize=(width, height*total_columns),
            sharex=True,
        )

        plt.subplots_adjust(hspace=hspace)

        for index, column in enumerate(self.column_names):
            ax = axs[index] if total_columns > 1 else axs
            ax.quiver(df.index, df[column], u, v, wind_speed, cmap=cmap, norm=norm,
                      scale=50, width=0.003, pivot='mid')

            ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.set_ylim([df[column].max() * -0.05, df[column].max() * 1.05])
            ax.set_xlim(df.first_valid_index(), df.last_valid_index())

            for label in ax.get_xticklabels(which='major'):
                label.set(rotation=30, horizontalalignment='right')

            ax.set_ylabel(column)

            cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
            cbar.set_label('Wind Speed (m/s)')

            ax.grid(True, linestyle='--', alpha=0.3)

        if save_plot:
            volcano_name = self.volcano_name if self.volcano_name is not None else 'volcano'
            start_date = self.start_date.strftime('%Y-%m-%d')
            end_date = self.end_date.strftime('%Y-%m-%d')
            filename = f'{prefix}_{volcano_name}_{start_date}-{end_date}' if filename is None else f'{prefix}_{filename}'

            figure_name = os.path.join(self.figures_dir, f'{filename}.png')
            print(f'📷 Saved to : {figure_name}')
            fig.savefig(figure_name, dpi=dpi)

        return fig, axs
