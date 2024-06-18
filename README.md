# magma-multigas
Python package for multigas sensor.

## Installation

---

Make sure you have at least Python 3.10 installed. You can install the package using `pip`
```python
pip install magma-multigas
```

## How to use

---

First we need to import the package.
```python
from magma_multigas import MultiGas
```

Then we need to add all the required files. Please change the files location with your multi gas data. At the moment I am not including the `span` data files. Will add later after having understanding what `span` data is.
```python
two_seconds = 'D:\\Projects\\magma-multigas\\input\\TANG_RTU_ChemData_Sec2.dat'
six_hours = 'D:\\Projects\\magma-multigas\\input\\TANG_RTU_Data_6Hr.dat'
one_minute = 'D:\\Projects\\magma-multigas\\input\\TANG_RTU_Wx_Min1.dat'
zero = 'D:\\Projects\\magma-multigas\\input\\TANG_RTU_Zero_Data.dat'
```

Initiating the MultiGas module with the code below. This code will correct the "NAN" data, and create a new file into your current poject directory. The default location is `<your project directory>/output/normalize` . 
```python
multigas = MultiGas(
    two_seconds=two_seconds,
    six_hours=six_hours,
    one_minute=one_minute,
    zero=zero
)
```

By initiating the module, we already have You can also check the output after run the code.
```markdown
💾 New file saved to D:\Projects\magma-multigas\output\normalize\TANG_RTU_ChemData_Sec2.dat
💾 New file saved to D:\Projects\magma-multigas\output\normalize\TANG_RTU_Data_6Hr.dat
💾 New file saved to D:\Projects\magma-multigas\output\normalize\TANG_RTU_Wx_Min1.dat
💾 New file saved to D:\Projects\magma-multigas\output\normalize\TANG_RTU_Zero_Data.dat
```

## Slicing and export for specific time

---

We can do filtering all the data we have by using this code. At this example I am trying to select ALL the data between `start_date = 2024-05-17` and `end_date = 2024-06-18`
```python
data_filtered = multigas.where_date_between(start_date='2024-05-17', end_date='2024-06-18').save(file_type='excel')
```

We can also save the filtering results. Only `excel`,`xls`,`xlsx` and `csv` are supported.
```python
data_filtered.save(file_type='excel')
```

All files would be saved into `<your project directory>/output/<file_type>`. You can also check the save location after run the `save()` command.
```markdown
✅ Data saved to: D:\Projects\magma-multigas\output\excel\two_seconds_2024-05-17_2024-06-18_TANG_RTU_ChemData_Sec2.xlsx
✅ Data saved to: D:\Projects\magma-multigas\output\excel\six_hours_2024-05-17_2024-06-18_TANG_RTU_Data_6Hr.xlsx
✅ Data saved to: D:\Projects\magma-multigas\output\excel\one_minute_2024-05-17_2024-06-18_TANG_RTU_Wx_Min1.xlsx
✅ Data saved to: D:\Projects\magma-multigas\output\excel\zero_2024-05-17_2024-06-18_TANG_RTU_Zero_Data.xlsx
```

## Selecting Data

---

As we know, we only have 4 type of data at the moment.
1. `two_seconds`
2. `six_hours`
3. `one_minute`
4. `zero`

To working on specific dataset, we can do it like this. To choose `two_secons` data:
```python
two_seconds_data = multigas.two_seconds
```
or:
```python
two_seconds_data = multigas.select('two_seconds').get()
```

You can change the `two_seconds` parameter with the available options above.

## Data Preview

---

After we select the data, we can do a quick view by typing this code:
```python
two_seconds_data.df
```
For anyone not familiar with `df` abbreviation, it is stand for _dataframe_. Just imagine it as an excel with header, but it is in python.

![df-review.png](images/df-review.png)

You can also see the columns name:
```python
two_seconds_data.columns
```
It will show all the columns name:
![columns.png](images/columns.png)

## Filtering
We can do fluent filtering data by using this code below. And it also support chaining filtering.  
```python
filtered_two_seconds = (two_seconds_data
                        .select_columns(column_names=['H2O','CO2','SO2','H2S','S_total'])
                        .where_date_between(start_date='2024-06-12', end_date='2024-06-18')
                        .where('Status_Flag', '==', 0)
                        .where_values_between(column_name='SO2', start_value=-0.129, end_value=-0.127)
                        .where_values_between(column_name='H2O', start_value=-260, end_value=-228)
                       )
```
We can read the above code as is:
> By using `two_seconds_data`, I want to select specific columns, such as: `H2O, CO2, SO2, S_Total` where the `Status_Flag` should have a `0` value. And the `SO2` columns must be between `-0.129` and `-0.127`. In addition I also want to filter `H2O` values between `-260` and `-228`.

To see the results, please run the `get()` method:
```python
filtered_two_seconds.get()
```

You can see the example of the results below. Here we only found 1 result based on our query filtering.
![filtering.png](images/filtering.png)

To check and count the results:
```python
filtered_two_seconds.count()
```

## Save Filtering Results

---

And we can save it, using `save_as()` method:
```python
filtered_two_seconds.save_as(file_type='excel')
```

You will get the information where your file is saved. By default, it should be in your `output` directory. Here is the example of the output:
```markdown
✅ Data saved to: D:\Projects\magma-multigas\output\excel\two_seconds_2024-06-16_2024-06-16_TANG_RTU_ChemData_Sec2.xlsx
```

## Plot

---

This package provide some basic functionality to plot some data. For the simplicity, we will use `six_hours` data as an example. We will do all the basic things above, from load, selecting, and filtering.

#### Selecting Six Hours
```python
six_hours_data = multigas.select('six_hours').get()
```

#### Date Filtering
```python
filtered_six_hours = six_hours_data.where_date_between(start_date='2024-05-17', end_date='2024-06-18')
```

#### Plot Initiating
Using `plot()` method to initiate
```python
plot_six_hours = filtered_six_hours.plot()
```

#### Plot Avg. CO2, H2S, and SO2
This package has some default plotting method. We will use `plot_co2_so2_h2s()` as an example:
```python
plot_six_hours.plot_co2_so2_h2s()
```

You can see the result here:
![plot_example_1.png](images/plot_example_1.png)

From the plot above we can see there ae some anomaly values (below 400). We can re-filtering once again to optimize the result. In this case we will select column `Avg_CO2_lowpass` which value greater than or equal to `250`
```python
filtered_six_hours = filtered_six_hours.where('Avg_CO2_lowpass', '>=', 250)
```

Then plot it:
```python
filtered_six_hours.plot().plot_co2_so2_h2s()
```

Result:
![plot_example_2.png](images/plot_example_2.png)

We can also plot as an individual plot by adding parameter `plot_as_individual=True`
```python
filtered_six_hours.plot().plot_co2_so2_h2s(
    plot_as_individual=True
)
```

And the result:
![plot_example_3.png](images/plot_example_3.png)

###  Plot Specific Column(s)
Select columns to plot:
```python
columns_to_plot = ['Avg_Wind_Speed', 'Avg_Wind_Direction', 'Avg_H2O', 'Avg_CO2_lowpass']
```

Then plot it:
```python
filtered_six_hours.plot(height=2).plot_columns(columns_to_plot)
```

Results:
![plot_example_4.png](images/plot_example_4.png)