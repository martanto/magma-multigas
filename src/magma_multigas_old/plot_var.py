from typing import Self, Tuple
from magma_var.plot import Plot as PlotVar
from .multigas_data import MultiGasData
import os
import matplotlib.pyplot as plt


class PlotWithMagma:
    def __init__(self, token: str, volcano_code: str, csv: str, type_of_data: str = 'six_hours',
                 start_date: str = None, end_date: str = None, overwrite: bool = False,
                 output_dir: str = None, verbose: bool = False):
        assert type_of_data in ('six_hours', 'two_seconds', 'one_minute'), \
            f"❌ type of data : {type_of_data} must be one of 'six_hours', 'two_seconds', 'one_minute'"

        assert os.path.exists(csv), f"❌ CSV : {csv} does not exist"

        self.token = token
        self.volcano_code = volcano_code
        self.csv = csv
        self.type_of_data = type_of_data
        multigas = MultiGasData(type_of_data=type_of_data, csv_file=csv, force=True)

        self.start_date = start_date if start_date is not None else multigas.start_date
        self.end_date = end_date if end_date is not None else multigas.end_date
        self.multigas = multigas
        self.multigas_df = multigas.df
        self.overwrite = overwrite
        self.output_dir = output_dir if output_dir is not None else multigas.output_dir
        self.verbose = verbose

        print(f'🌋 Volcano Code: {volcano_code}')
        print(f'ℹ️ Using {type_of_data} data with file : {csv}')
        print(f'⌚ Start date : {self.start_date}')
        print(f'⌚ End date : {self.end_date}')
        print(f'📁 Output Directory : {self.output_dir}')
        print(f'🗃️ Overwrite : {overwrite}')
        print('-' * 60)
        print(f'⌛ Downloading VAR from MAGMA ....')

        plot_var = PlotVar(
            token=token,
            volcano_code=volcano_code,
            start_date=self.start_date,
            end_date=self.end_date,
            earthquake_code=['*'],
            current_dir=self.output_dir,
            verbose=False,
        )

        if len(plot_var.df) == 0:
            raise ValueError(f'❌ VAR not found for {volcano_code} from {self.start_date} to {self.end_date}')

        self.plot_var = plot_var
        self.var_df = plot_var.df

        print(f'✅ VAR Downloaded successfully for {volcano_code} from {self.start_date} to {self.end_date}')

    def show_var(self, interval=14, title='Volcanic Activity Report', figsize=(10, 1), title_fontsize=12,
                 title_y_location=0.96, figure_ylabel_fontsize=9, figure_ylabel_x=0.05,
                 x_labelsize=8, y_labelsize=8, ) -> Tuple[plt.Figure, plt.Axes]:
        return self.plot_var.show(
            interval=interval,
            title=title,
            figsize=figsize,
            title_fontsize=title_fontsize,
            title_y_location=title_y_location,
            figure_ylabel_fontsize=figure_ylabel_fontsize,
            figure_ylabel_x=figure_ylabel_x,
            x_labelsize=x_labelsize,
            y_labelsize=y_labelsize,
        )

    def run(self) -> Self:
        return self
