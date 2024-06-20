import os
import pandas as pd

from .query import Query
from .validator import validate_file_type
from .plot import Plot
from pathlib import Path
from typing import Dict, Tuple


def start_and_end_date(df: pd.DataFrame = None) -> Tuple[str, str]:
    """Return start and end date from filtered dataframe

    Returns:
        Tuple[str, str]: start and end date from filtered dataframe
    """
    return (df.index[0].strftime('%Y-%m-%d'),
            df.index[-1].strftime('%Y-%m-%d'))


class MultiGasData(Query):
    def __init__(self, type_of_data: str, csv_file: str, force: bool = False):
        """Data of MultiGas
        """
        self.current_dir = os.getcwd()
        output_dir = os.path.join(self.current_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        self.force = force
        self.output_dir = output_dir
        self.csv_file: str = self.replace_nan(csv_file)
        self.filename: str = Path(self.csv_file).stem
        self.type_of_data: str = type_of_data

        super().__init__(self.set_df())
        print(f"📅 {type_of_data} available from: {self.start_datetime} to {self.end_datetime}")

    def __str__(self) -> str:
        """Return type of multigas data"""
        return self.type_of_data

    def __getattr__(self, column_name: str) -> pd.DataFrame:
        """Get dataframe, when columns is not available"""
        return self.df_original[column_name]

    def __repr__(self) -> str:
        """Class representative"""
        return self.describe()

    def describe(self) -> str:
        """Describe class"""
        return (f"{type(self).__name__}(type_of_data={self.type_of_data}, length={self.count()}, "
                f"start_date={self.start_datetime}, end_date={self.end_datetime})")

    def data(self) -> pd.DataFrame:
        """Alias for df

        Returns:
            pd.DataFrame
        """
        return self.df

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
        csv = self.csv_file

        with open(csv, 'r') as file:
            contents: list[str] = file.readlines()[0].replace("\"", '').split(',')
            headers: dict[str, str] = {
                "format_data": contents[0].strip(),
                'station': contents[1].strip(),
                'logger_type': contents[2].strip(),
                'data_counts': len(self.df_original),
                'firmware': contents[4].strip(),
                'program_name': contents[5].strip(),
                'unknown': contents[6].strip(),
                'file_sampling': contents[7].strip(),
            }
            file.close()
            return headers

    def set_df(self, csv_file: str = None, index_col: str = None) -> pd.DataFrame:
        """Get data from MultiGas

        Returns:
            pd.DataFrame: data from MultiGas
        """
        if csv_file is None:
            csv_file = self.csv_file

        if index_col is None:
            index_col = 'TIMESTAMP'

        df = pd.read_csv(csv_file,
                         skiprows=lambda x: x in [0, 2, 3],
                         parse_dates=[index_col],
                         index_col=[index_col])
        return df

    def save_as(self, file_type: str = 'excel', output_dir: str = None,
                filename: str = None, use_filtered: bool = True, **kwargs) -> str | None:
        """Save data from MultiGas to specified file type

        Args:
            file_type (str): Chose between 'csv', 'excel', 'xlsx', 'xls'
            output_dir (str): directory to save to
            filename (str): filename
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

        df = self.get() if use_filtered else self.df_original

        start_date, end_date = start_and_end_date(df)

        if filename is None:
            filename = f"{self.type_of_data}_{start_date}_{end_date}_{self.filename}.{file_extension}"

        file_location: str = os.path.join(output_dir, f"{filename}.{file_extension}")

        if not df.empty:
            df.to_excel(file_location, **kwargs) if file_type != 'csv' \
                else df.to_csv(file_location, **kwargs)
            print(f'✅ Data saved to: {file_location}')
            return file_location
        print(f'⚠️ Data {self.filename} is empty. Skip.')
        return None

    def plot(self, y_min: float = None, y_max: float = None,
             width: int = 12, height: int = 4) -> Plot:
        """Plot selected data and columns.

        Args:
            y_min (float): Minimum value
            y_max (float): Maximum value
            width (int): Figure width
            height (int): Figure height

        Returns:
            save_path (str): Path to save plot
        """
        return Plot(
            df=self.get(),
            y_min=y_min,
            y_max=y_max,
            width=width,
            height=height,
        )
