from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots, savefig, show
from ds_charts import get_variable_types, HEIGHT
from matplotlib.pyplot import figure, savefig, show, title
from seaborn import heatmap

register_matplotlib_converters()
filename = "lab01_data_profiling/data/set1_NYC_collisions_tabular.csv"
data = read_csv(filename, index_col='CRASH_DATE', parse_dates=True, infer_datetime_format=True)

corr_mtx = data.corr()
print(corr_mtx)

fig = figure(figsize=[12, 12])

heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
title('Correlation analysis')
savefig(f'lab01_data_profiling/images/correlation_analysis_set1.png')
show()