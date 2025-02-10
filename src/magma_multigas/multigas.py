from .multigas_data import MultiGasData
from .plot import Plot
from .validator import validate_selected_data, validate_file_type
from typing import Any, Dict, List, Self

types = ('two_seconds', 'six_hours', 'one_minute', 'zero', 'span')

info = """
ℹ️ =============================
- Six hours of data used to plot:
1. CO2 - SO2 - H2S Concentration
2. CO2/H2S Ratio
3. H2O/CO2 Ratio
4. H2S/SO2 Ratio
5. CO2/SO2 Ratio
6. CO2/Stotal Ratio,
7. Sulfur Speciation (%) = Avg_SO2_proportion, Avg_H2S_proportion

- One minute of data used to plot:
1. Air Temperature
2. Air Humidity
3. Battery Voltage
4. Wind Speed
5. Wind Direction

- Zero data used to plot:
1. CO2 - Zero
2. H2S - Zero
3. SO2 - Zero
"""


class MultiGas:
    def __init__(self,
                 six_hours: str,
                 one_minute: str,
                 zero: str,
                 two_seconds: str = None,
                 span: str = None,
                 overwrite: bool = False,
                 default: str = 'six_hours',
                 normalize_dir: str = None,
                 data_length: int = None,):

        if overwrite is True:
            print(f"⚠️ Existing data will be overwritten.")

        self.files: Dict[str, str] = {
            'two_seconds': two_seconds,
            'six_hours': six_hours,
            'one_minute': one_minute,
            'zero': zero,
            'span': span
        }

        self.two_seconds: MultiGasData | None = MultiGasData('two_seconds',
                                                             two_seconds,
                                                             force=overwrite,
                                                             normalize_dir=normalize_dir) if two_seconds is not None else None
        self.six_hours: MultiGasData | None = MultiGasData('six_hours',
                                                           six_hours,
                                                           force=overwrite,
                                                           normalize_dir=normalize_dir)
        self.one_minute: MultiGasData | None = MultiGasData('one_minute',
                                                            one_minute,
                                                            force=overwrite,
                                                            normalize_dir=normalize_dir,
                                                            data_length=data_length)
        self.zero: MultiGasData | None = MultiGasData('zero',
                                                      zero, force=overwrite,
                                                      normalize_dir=normalize_dir)
        self.span: MultiGasData | None = MultiGasData('span',
                                                      span,
                                                      force=overwrite,
                                                      normalize_dir=normalize_dir) if span is not None else None

        self.data_selected: str = default

        print(f'ℹ️ DEFAULT selected: {default}')
        self.selected: MultiGasData = self.get(default)

    def __repr__(self) -> str:
        """Class representative"""
        return (f"{type(self).__name__}(two_seconds={type(self.two_seconds)}, six_hours={type(self.six_hours)}, "
                f"one_minute={type(self.one_minute)}, zero={type(self.zero)}, span={type(self.span)})")

    @str
    def info(self):
        """MultiGas Information"""
        print(info)

    def select(self, type_of_data: str) -> Self:
        """Select data based period of measurement

        Args:
            type_of_data: Type of data. Choose 'two_seconds', 'six_hours', 'one_minute', 'zero', 'span'

        Returns:
            MultiGasData: Selected data
        """
        type_of_data = type_of_data.lower()

        validate_selected_data(type_of_data)

        if type_of_data not in types:
            raise ValueError(f'⛔ Type of data must be one of {types}')

        self.data_selected = type_of_data

        self.selected: MultiGasData = self.get(type_of_data)

        print("ℹ️ {} data selected.".format(type_of_data))
        return self

    def where_date_between(self, start_date: str, end_date: str) -> Self:
        """Filtering ALL data between start and end date.

        Args:
            start_date (str): Optional. Start date. Date format YYYY-MM-DD
            end_date (str): Optional. End date. Date format YYYY-MM-DD

        Returns:
            Self: MultiGas
        """
        for type_of_data in types:
            multigas_data: MultiGasData = self.get(type_of_data)
            if multigas_data is not None:
                multigas_data.where_date_between(start_date, end_date).get()

        return self

    @property
    def columns(self) -> list[str]:
        """Get columns for specific data type.

        Returns:
            list[str]: List of column names
        """
        print("List of columns for {} data".format(self.data_selected))
        return self.selected.columns

    def plot(self, width: int = 12, height: int = 4, y_max_multiplier: float = 1.0,
             datetime_column: str = 'TIMESTAMP', margins: float = 0.05, style = "whitegrid") -> Plot:
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
        print("Data selected to plot: {}".format(self.data_selected))

        return Plot(
            df=self.selected.df,
            width=width,
            height=height,
            datetime_column=datetime_column,
            y_max_multiplier=y_max_multiplier,
            margins=margins,
            style=style
        )

    def get(self, type_of_data: str = None) -> MultiGasData:
        """Get selected data.

        Returns:
            MultiGasData: Selected data
        """
        if type_of_data is None:
            type_of_data = self.data_selected

        print("✅ {} data loaded".format(type_of_data))

        match type_of_data:
            case 'two_seconds':
                return self.two_seconds
            case 'six_hours':
                return self.six_hours
            case 'one_minute':
                return self.one_minute
            case 'zero':
                return self.zero
            case 'span':
                return self.span

    def extract_daily(self, directory_name: str) -> Dict[str, List[Dict[str, int]]]:
        """Extract daily data.

        Args:
            directory_name (str): Where available data would be saved.

        Returns:
            Self: MultiGas
        """
        availability: Dict[str, List[Dict[str, int]]] = {}

        for type_of_data in self.files.keys():
            multigas_data: MultiGasData | None = self.select(type_of_data).get()
            if multigas_data is not None:
                availability[type_of_data] = multigas_data.extract_daily(directory_name)

        return availability

    def save(self, file_type: str = 'excel', output_dir: str = None,
             use_filtered: bool = True, **kwargs) -> None:
        """Save ALL fixed data as an excel file and/or CSV.

        Args:
            file_type (str): Chose between 'csv', 'excel', 'xlsx', 'xls'
            output_dir (str): Output directory. Use default output directory if None.
            use_filtered (bool): use filtered data. Defaults to True
            kwargs (dict): Keyword arguments
        """
        validate_file_type(file_type)

        params = {
            'output_dir': output_dir,
            'use_filtered': use_filtered,
        }

        for type_of_data in types:
            multigas_data: MultiGasData | None = self.get(type_of_data)
            if multigas_data is not None:
                multigas_data.save_as(file_type=file_type, **params, **kwargs)
            else:
                print(f'⚠️ Data {type_of_data} is empty. Skip.')
