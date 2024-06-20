from .multigas_data import MultiGasData
from .plot import Plot
from .validator import validate_selected_data, validate_file_type
from typing import Dict, Self

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
                 two_seconds: str,
                 six_hours: str,
                 one_minute: str,
                 zero: str,
                 span: str = None,
                 overwrite: bool = False,
                 default: str = 'six_hours'):

        if overwrite is True:
            print(f"⚠️ Existing data will be overwritten.")

        self.files: Dict[str, str] = {
            'two_seconds': two_seconds,
            'six_hours': six_hours,
            'one_minute': one_minute,
            'zero': zero,
            'span': span
        }

        self.two_seconds: MultiGasData = MultiGasData('two_seconds', two_seconds, force=overwrite)
        self.six_hours: MultiGasData = MultiGasData('six_hours', six_hours, force=overwrite)
        self.one_minute: MultiGasData = MultiGasData('one_minute', one_minute, force=overwrite)
        self.zero: MultiGasData = MultiGasData('zero', zero, force=overwrite)
        self.span: MultiGasData | None = MultiGasData('span', span, force=overwrite) \
            if span is not None else None

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
        print("Data selected to plot: {}".format(self.data_selected))

        return Plot(
            df=self.selected.df,
            y_min=y_min,
            y_max=y_max,
            width=width,
            height=height,
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
