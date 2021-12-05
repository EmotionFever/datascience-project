from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots, savefig, show
from ds_charts import get_variable_types, HEIGHT

register_matplotlib_converters()
#filename = 'lab01_data_profiling\data\set2_air_quality_tabular.csv'
filename = 'lab01_data_profiling\data\set1_NYC_collisions_tabular.csv'
data = read_csv(filename, index_col='CRASH_DATE', parse_dates=True, infer_datetime_format=True)

print(data.head())

numeric_vars = get_variable_types(data)['Numeric']
print(numeric_vars)
numeric_vars = ['CRASH_TIME', 'PERSON_AGE', 'VEHICLE_ID']
not_working = ['CRASH_DATE']
#these now contain only person age and vehicle id, is that correct?
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

rows, cols = len(numeric_vars)-1, len(numeric_vars)-1
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(len(numeric_vars)):
    var1 = numeric_vars[i]
    for j in range(i+1, len(numeric_vars)):
        var2 = numeric_vars[j]
        axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1], data[var2])
savefig('lab01_data_profiling\images\sparsity_study_numeric_set1.png')
show()

from matplotlib.pyplot import savefig, show, subplots
from ds_charts import HEIGHT, get_variable_types

#code error because of nans, give them string
data.fillna('A',inplace=True)

symbolic_vars = get_variable_types(data)['Symbolic']
print(symbolic_vars)
symbolic_vars = ['BODILY_INJURY', 'SAFETY_EQUIPMENT', 'PERSON_SEX', 'PERSON_TYPE', 'PED_LOCATION', 'CONTRIBUTING_FACTOR_2', 'EJECTION', 'COMPLAINT', 'EMOTIONAL_STATUS', 'CONTRIBUTING_FACTOR_1', 'POSITION_IN_VEHICLE', 'PED_ROLE', 'PED_ACTION']
#symbolic_vars = ['BODILY_INJURY',  'PERSON_SEX', 'PERSON_TYPE', 'PED_ACTION']
not_working = ['SAFETY_EQUIPMENT',  'PED_LOCATION']
if [] == symbolic_vars:
    raise ValueError('There are no symbolic variables.')

rows, cols = len(symbolic_vars)-1, len(symbolic_vars)-1
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
for i in range(len(symbolic_vars)):
    var1 = symbolic_vars[i]
    for j in range(i+1, len(symbolic_vars)):
        var2 = symbolic_vars[j]
        axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1], data[var2])
savefig(f'lab01_data_profiling\images\sparsity_study_symbolic_set1.png')
show()