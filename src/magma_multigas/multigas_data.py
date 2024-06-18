import os
import pandas as pd

from .query import Query
from .validator import validate_file_type
from .plot import Plot
from pathlib import Path
from typing import Dict, Self, Tuple


def start_and_end_date(df: pd.DataFrame = None) -> Tuple[str, str]:
    """Return start and end date from filtered dataframe

    Returns:
        Tuple[str, str]: start and end date from filtered dataframe
    """
    return (df.index[0].strftime('%Y-%m-%d'),
            df.index[-1].strftime('%Y-%m-%d'))


class MultiGasData(Query):
    def __init__(self, type_of_data: str, csv: str, force: bool = False):
        """Data of MultiGas
        """
        self.current_dir = os.getcwd()
        output_dir = os.path.join(self.current_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        self.force = force
        self.output_dir = output_dir
        self.csv: str = self.replace_nan(csv)
        self.filename: str = Path(self.csv).stem
        self.type_of_data: str = type_of_data
        self.original_data: pd.DataFrame = self._df()
        self.filtered_data: pd.DataFrame = self.original_data.copy()
        super().__init__(self.filtered_data)

    def is_filtered(self) -> bool:
        """Check if data is filtered or not

        Returns:
            True if filtered.
        """
        return False if self.original_data.equals(self.filtered_data) else True

    def replace_nan(self, csv: str) -> str:
        """Replacing 'NAN' value with np.NaN

        Args:
            csv (str): csv file path

        Returns:
            str: csv file path location
        """
        csv_dir, csv_filename = os.path.split(csv)

        normalize_dir = os.path.join(self.output_dir, 'normalize')
        os.makedirs(normalize_dir, exist_ok=True)

        save_path = os.path.join(normalize_dir, csv_filename)

        if os.path.isfile(save_path) and not self.force:
            print(f"✅ File already exists : {save_path}")
            return save_path

        with open(csv, 'r') as file:
            file_content: str = file.read()
            new_content = file_content.replace("\"NAN\"", "")
            file.close()
            with open(save_path, 'w') as new_file:
                new_file.write(new_content)
                print(f"💾 New file saved to {save_path}")
                new_file.close()
                return new_file.name

    @property
    def metadata(self) -> Dict[str, str]:
        """Metadata property of MultiGas

        Returns:
            Dict[str, str]: metadata property of MultiGas
        """
        csv = self.csv

        with open(csv, 'r') as file:
            contents: list[str] = file.readlines()[0].replace("\"", '').split(',')
            headers: dict[str, str] = {
                "format_data": contents[0].strip(),
                'station': contents[1].strip(),
                'logger_type': contents[2].strip(),
                'data_counts': len(self.original_data),
                'firmware': contents[4].strip(),
                'program_name': contents[5].strip(),
                'unknown': contents[6].strip(),
                'file_sampling': contents[7].strip(),
            }
            file.close()
            return headers

    def refresh(self) -> Self:
        """Refresh original values and filtered values"""
        self.original_data = self._df()
        self.filtered_data = self.original_data.copy()
        super().__init__(self.filtered_data)
        return self

    def _df(self) -> pd.DataFrame:
        """Get data from MultiGas

        Returns:
            pd.DataFrame: data from MultiGas
        """
        df = pd.read_csv(self.csv,
                         skiprows=lambda x: x in [0, 2, 3],
                         parse_dates=['TIMESTAMP'],
                         index_col=['TIMESTAMP'])
        return df

    def save_as(self, file_type: str = 'excel', output_dir: str = None,
                use_filtered: bool = True, **kwargs) -> str | None:
        """Save data from MultiGas to specified file type

        Args:
            file_type (str): Chose between 'csv', 'excel', 'xlsx', 'xls'
            output_dir (str): directory to save to
            use_filtered (bool): use filtered data
            kwargs (dict): keyword arguments

        Returns:
            File save location. Return None if data is empty
        """
        validate_file_type(file_type)

        file_extension = 'csv'
        sub_output_dir = 'csv'

        if file_type != 'csv':
            file_extension = 'xlsx'
            sub_output_dir = 'excel'

        if output_dir is None:
            output_dir = self.output_dir

        output_dir = os.path.join(output_dir, sub_output_dir)
        os.makedirs(output_dir, exist_ok=True)

        df = self.get() if use_filtered else self.original_data

        start_date, end_date = start_and_end_date(df)
        filename = f"{start_date}_{end_date}_{self.filename}.{file_extension}"
        file_location: str = os.path.join(output_dir, filename)

        if not df.empty:
            df.to_excel(file_location, **kwargs) if file_type != 'csv' \
                else df.to_csv(file_location, **kwargs)
            print(f'✅ Data saved to: {file_location}')
            return file_location
        print(f'⚠️ Data {self.filename} is empty. Skip.')
        return None

    def plot(self, y_min: float = None, y_max: float = None, y_max_multiplier: float = 1,
             width: int = 12, height: int = 4) -> Plot:
        """Plot selected data and columns.

        Args:
            y_min (float): Minimum value
            y_max (float): Maximum value
            y_max_multiplier (float): Multiplier factor
            width (int): Figure width
            height (int): Figure height

        Returns:
            save_path (str): Path to save plot
        """
        return Plot(
            df=self.get(),
            y_min=y_min,
            y_max=y_max,
            y_max_multiplier=y_max_multiplier,
            width=width,
            height=height,
        )

    def get(self) -> pd.DataFrame:
        self.filtered_data: pd.DataFrame = super().get()
        return self.filtered_data
