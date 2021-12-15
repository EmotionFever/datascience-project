from pandas import read_csv, concat, DataFrame
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import savefig, show, subplots, figure, Axes
from ds_charts import get_variable_types, choose_grid, HEIGHT, multiple_line_chart, bar_chart
from numpy import nan

register_matplotlib_converters()
file = 'set1'
#filename = 'data/set1_NYC_collisions_tabular.csv'
filename = 'lab02_data_preparation\ew_data\set1_symbolic_to_numeric.csv'
data = read_csv(filename)
data.describe(include='all')

#Charts with details about single numeric
# numeric_vars = get_variable_types(data)['Numeric']
numeric_vars = ['PERSON_AGE']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')
rows, cols = choose_grid(len(numeric_vars))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
    axs[i, j].boxplot(data[numeric_vars[n]].dropna().values)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
#savefig('images/set1_outlier_imputation_PERSON_AGE_new.png')
show()

# replace with NaN values > 110 or < 0 PERSON_AGE
data['PERSON_AGE'] = data['PERSON_AGE'].mask(data['PERSON_AGE'].lt(0),nan)
data['PERSON_AGE'] = data['PERSON_AGE'].mask(data['PERSON_AGE'].gt(110),nan)

data.describe(include='all')

#Charts with details about single numeric
# numeric_vars = get_variable_types(data)['Numeric']
numeric_vars = ['PERSON_AGE']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')
rows, cols = choose_grid(len(numeric_vars))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
    axs[i, j].boxplot(data[numeric_vars[n]].dropna().values)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
#savefig('images/set1_outlier_imputation_PERSON_AGE_old.png')
show()

data.to_csv(f'lab02_data_preparation\ew_data\{file}_outliers.csv', index=False)
