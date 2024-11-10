from .plotly_calplot import calplot
from typing import Dict, List, Self
import plotly.express as px
from plotly import graph_objects as go
import pandas as pd
import os


class PlotAvailability:
    def __init__(self, csv_availability: str, title: str):
        self._df: pd.DataFrame = pd.read_csv(csv_availability)
        self.title: str = title
        self.start_date: pd.Timestamp = self.df['date'].min()
        self.end_date: pd.Timestamp = self.df['date'].max()

    @property
    def unique_years(self) -> List[int]:
        """Get list of unique years

        Returns:
            List[int]
        """
        return list(self.df['date'].dt.year.unique())

    @property
    def df(self) -> pd.DataFrame:
        """Get plot dataframe

        Returns:
            pd.DataFrame
        """
        df = self._df
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(by=['date'], inplace=True)
        return df

    @property
    def output_dir(self) -> str:
        """Output directory

        Returns:
            str
        """
        output_dir: str = os.path.join(os.getcwd(), 'output', 'figures')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @property
    def figure(self) -> go.Figure:
        """Plotly graph_object figure

        Returns:
            go.Figure
        """
        title = self.title

        figure: go.Figure = calplot(
            data=self.df,
            x="date",
            y="percentage_available",
            colorscale=px.colors.diverging.RdYlGn,
        ).update_layout(
            self.axis,
            title=title,
        ).update_traces(
            showscale=True,
            selector=dict(type='heatmap'),
            zmax=100,
            zmin=0,
        )

        return figure

    @property
    def axis(self) -> Dict[str, Dict[str, str]]:
        """Update all axis values

        Returns:
            Dict[str, Dict[str, str]]
        """
        axes = self.yaxis
        axes.update(self.xaxis)

        return axes

    @property
    def yaxis(self) -> Dict[str, Dict[str, str]]:
        """Set label for yaxis

        Returns:
            Dict[str, Dict[str, str]]
        """
        years = self.unique_years
        y_axes = {}

        for index, year in enumerate(years):
            key = "yaxis" if index == 0 else f"yaxis{index + 1}"
            y_axes[key] = {}
            y_axes[key]['title'] = f"{year}"

        return y_axes

    @property
    def xaxis(self) -> Dict[str, Dict[str, str]]:
        """Xaxis ticktext values

        Returns:
            Dict[str, Dict[str, str]]
        """
        years = self.unique_years
        ticktext = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                    'Sept', 'Okt', 'Nov', 'Dec']
        x_axes = {}

        for index, year in enumerate(years):
            key = "xaxis" if index == 0 else f"xaxis{index + 1}"
            x_axes[key] = {}
            x_axes[key]['ticktext'] = ticktext

        return x_axes

    def filter(self, start_date: str, end_date: str) -> Self:
        """Filter data by start and end dates.

        Args:
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.

        Returns:
            Self
        """
        df = self.df
        self._df = df[(df['date'] > start_date) & (df['date'] < end_date)]
        self.start_date = start_date
        self.end_date = end_date
        return self

    def show(self) -> None:
        """Show figure

        Returns:
            None
        """
        self.figure.show()

    def save(self, filename: str = None, filetype: str = 'jpg') -> bool:
        """Save figure

        Args:
            filename: Filename
            filetype: File type

        Returns:
            bool
        """
        if filename is None:
            start_date = self.start_date.strftime('%Y-%m-%d')
            end_date = self.end_date.strftime('%Y-%m-%d')
            filename = f"availability_{self.title}_{start_date}-{end_date}.{filetype}"

        filepath = os.path.join(self.output_dir, filename)

        try:
            self.figure.write_image(filepath)
            print(f"🏞️ Saved to: {filepath}")
            return True
        except Exception as e:
            print(f"⛔ {e}")
            return False
