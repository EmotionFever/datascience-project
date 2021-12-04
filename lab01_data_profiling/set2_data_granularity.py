# This code
# Make a choice of bins or adjust scale
# What is the goal of this granularity thing --> find out! 
# Write report



from pandas import read_csv

filename = 'lab01_data_profiling\data\set2_air_quality_tabular.csv'
data = read_csv(filename)

values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}

from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import subplots, savefig, show

variables = get_variable_types(data)['Numeric']
if [] == variables:
    raise ValueError('There are no numeric variables.')

rows, cols = choose_grid(len(variables))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT))
i, j = 0, 0
for n in range(len(variables)):
    axs[i, j].set_title('Histogram for %s'%variables[n])
    axs[i, j].set_xlabel(variables[n])
    axs[i, j].set_ylabel('nr records')
    axs[i, j].hist(data[variables[n]].values, bins=1000)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('lab01_data_profiling\images\granularity_single_set2.png')
show()