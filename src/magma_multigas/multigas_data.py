import os
import pandas as pd

from .query import Query
from .validator import validate_file_type
from .plot import Plot
from pathlib import Path
from typing import Any, Dict, List, Tuple


def start_and_end_date(df: pd.DataFrame = None) -> Tuple[str, str]:
    """Return start and end date from filtered dataframe

    Returns:
        Tuple[str, str]: start and end date from filtered dataframe
    """
    return (df.index[0].strftime('%Y-%m-%d'),
            df.index[-1].strftime('%Y-%m-%d'))


class MultiGasData(Query):
    total_data = {
        'two_seconds': 5760,
        'one_minute': 1440,
        'six_hours': 4,
        'zero': 4,
    }

    def __init__(self, type_of_data: str,
                 csv_file: str,
                 force: bool = False,
                 index_col: str = None,
                 normalize_dir: str = None,
                 data_length: int = None):
        """Data of MultiGas
        """
        self.current_dir = os.getcwd()
        output_dir = os.path.join(self.current_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        self.force = force
        self.index_col = index_col if index_col is not None else 'TIMESTAMP'
        self.normalize_dir = normalize_dir
        self.data_length = data_length
        self.output_dir = output_dir
        self.csv_file: str = self.replace_nan(csv_file)
        self.filename: str = Path(self.csv_file).stem
        self.type_of_data: str = type_of_data

        super().__init__(self.set_df(csv_file=self.csv_file, index_col=index_col))
        print(f"📅 {type_of_data} available from: {self.start_datetime} to {self.end_datetime}")

        # Removing duplicates
        self.df.drop_duplicates(inplace=True)

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

        if self.normalize_dir is not None:
            normalize_dir = os.path.join(normalize_dir, self.normalize_dir)

        os.makedirs(normalize_dir, exist_ok=True)

        save_path = os.path.join(normalize_dir, csv_filename)

        if os.path.isfile(save_path) and not self.force:
            print(f"✅ File already exists : {save_path}")
            return save_path

        with open(csv, 'r') as file:
            file_content: str = file.read()
            new_content = file_content.replace("NAN", "").replace("\"NAN\"", "")
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
                'format_data': contents[0].strip(),
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

        try:
            df = pd.read_csv(csv_file,
                             skiprows=lambda x: x in [0, 2, 3],
                             parse_dates=[index_col],
                             index_col=[index_col])
            return df
        except Exception as e:
            print(f"⚠️ {e}. Parameters skiprows=[0,2,3] not working. Trying next read method.")

        try:
            df = pd.read_csv(csv_file,
                             parse_dates=[index_col],
                             index_col=[index_col])
            return df
        except Exception as e:
            print(f"❌ {e}")

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

        sub_output_dir = 'csv'

        if file_type != 'csv':
            sub_output_dir = 'excel'

        if output_dir is None:
            output_dir = self.output_dir

        output_dir = os.path.join(output_dir, self.metadata['station'], sub_output_dir)
        os.makedirs(output_dir, exist_ok=True)

        df = self.get() if use_filtered else self.df_original

        start_date, end_date = start_and_end_date(df)

        if filename is None:
            filename = f"{self.type_of_data}_{start_date}_{end_date}_{self.filename}"

        file_location: str = os.path.join(output_dir, f"{filename}")

        if not df.empty:
            df.to_excel(file_location, **kwargs) if file_type != 'csv' \
                else df.to_csv(file_location, **kwargs)
            print(f'✅ Data saved to: {file_location}')
            return file_location
        print(f'⚠️ Data {self.filename} is empty. Skip.')
        return None

    def _percentage_availability(self, total_data: int) -> float:
        _total_data = self.total_data[self.type_of_data]

        if self.data_length is not None:
            _total_data = self.data_length

        return total_data / _total_data * 100

    def extract_daily(self, directory_name: str,
                      file_type: str = 'csv',
                      save_availability: bool = True) -> List[Dict[str, int]]:
        """Extract daily data from MultiGas

        Args:
            directory_name (str): Where available data would be saved.
            file_type (str): Chose between 'csv' or 'xlsx'
            save_availability (bool): save availability data

        Returns:
            List[Dict[str, int]]: daily data from MultiGas
        """
        assert (file_type == 'csv' or file_type == 'xlsx'), f"{file_type} is not supported. Please use csv or xlsx"

        availability: List[Dict[str, Any]] = []

        output_dir: str = os.path.join(self.output_dir, 'daily', directory_name, self.type_of_data)
        os.makedirs(output_dir, exist_ok=True)

        df = self.df
        start_date, end_date = start_and_end_date(df)
        dates: pd.DatetimeIndex = pd.date_range(start_date, end_date, freq='D')

        for date_obj in dates:
            date_str: str = date_obj.strftime('%Y-%m-%d')
            filename: str = f"{date_str}.{file_type}"
            output_file: str = os.path.join(output_dir, filename)
            df_per_date: pd.DataFrame = df.loc[date_str]

            if df_per_date.empty is True:
                print(f"⚠️ {date_str} :: [{self.type_of_data}] Empty data.")
                date_availability: Dict[str, Any] = {
                    'date': date_str,
                    'total_data': 0,
                    'percentage_available': 0
                }
                availability.append(date_availability)
                continue

            if file_type == 'csv':
                df_per_date.to_csv(output_file, index=True)
            else:
                df_per_date.to_excel(output_file, index=True)

            date_availability: Dict[str, Any] = {
                'date': date_str,
                'total_data': len(df_per_date),
                'percentage_available': self._percentage_availability(total_data=len(df_per_date))
            }

            availability.append(date_availability)

            print(f"💾 {date_str} :: [{self.type_of_data}] Saved to {output_file}")

        if save_availability is True:
            availability_dir: str = os.path.join(self.output_dir, 'availability', directory_name, )
            os.makedirs(availability_dir, exist_ok=True)

            availability_file: str = os.path.join(availability_dir, f'{self.type_of_data}.csv')

            df_availability: pd.DataFrame = pd.DataFrame.from_records(availability)
            df_availability.to_csv(availability_file, index=False)

        return availability

    def plot(self, width: int = 12, height: int = 4, y_max_multiplier: float = 1.0,
             datetime_column: str = 'TIMESTAMP', margins: float = 0.05, style="whitegrid") -> Plot:
        """Plot selected data and columns.

        Args:
            width (int, optional): Width of the plot. Defaults to 12.
            height (int, optional): Height of the plot or column. Defaults to 4.
            datetime_column (str, optional): A column name containing the datetime. Defaults to 'TIMESTAMP'.
            y_max_multiplier (float, optional): Maximum multiplier for the y-axis. Defaults to 1.0.
            margins (float, optional): Margin of the plot. Defaults to 0.05
            style (str, optional): Style of the plot. Defaults to "whitegrid".
            
        Returns:
            Plot class
        """
        return Plot(
            df=self.get(),
            width=width,
            height=height,
            datetime_column=datetime_column,
            y_max_multiplier=y_max_multiplier,
            margins=margins,
            style=style
        )
