from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots, savefig, show
from ds_charts import get_variable_types, HEIGHT

register_matplotlib_converters()
filename = "lab01_data_profiling/data/set2_air_quality_tabular.csv"
data = read_csv(filename, index_col='date', parse_dates=True, infer_datetime_format=True)

#choose more numeric variables?
numeric_vars = get_variable_types(data)['Numeric']
print(numeric_vars)
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

rows, cols = len(numeric_vars)-1, len(numeric_vars)-1
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)

# Color per class
data['plot_class'] = data['ALARM']
data["plot_class"].replace({"Danger": "red", "Safe": "green"}, inplace=True)
plot_class = data["plot_class"].to_list()

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
        
        #plt.scatter(x, y, c=y, s=500, cmap='gray')
savefig('lab01_data_profiling\images\set2\sparsity_study_numeric_set2.png')
show()


for i in range(len(numeric_vars)):
    var1 = numeric_vars[i]
    for j in range(i+1, len(numeric_vars)):
        var2 = numeric_vars[j]
        axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1], data[var2],  c = plot_class)
savefig('lab01_data_profiling\images\set2\sparsity_study_numeric_set2.png')
show()

from matplotlib.pyplot import savefig, show, subplots
from ds_charts import HEIGHT, get_variable_types


# Handpick symbolic variables because Python choses the wrong ones
symbolic_vars = get_variable_types(data)['Symbolic']
#symbolic_vars = ['City_EN', 'Prov_EN', 'ALARM']

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
savefig('lab01_data_profiling\images\set2\sparsity_study_symbolic_set2.png')
show()


