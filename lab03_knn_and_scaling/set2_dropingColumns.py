from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, get_variable_types
from pandas import DataFrame

#Load data
register_matplotlib_converters()
file = 'set1'
filename = 'lab02_data_preparation/ew_data/set2_mv_dummified.csv'
data = read_csv(filename, na_values='', parse_dates=True, infer_datetime_format=True)

data = data.drop(labels=["GbCity", "City_EN", "Prov_EN", "Field_1"], axis=1)

data.to_csv('lab03_knn_and_scaling/ew_data/set2_ready_to_scale.csv', index=False)

