from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from pandas import Series
from matplotlib.pyplot import savefig, show, subplots, figure, Axes
from ds_charts import get_variable_types, multiple_bar_chart, choose_grid, HEIGHT, multiple_line_chart, bar_chart
from seaborn import distplot
from scipy.stats import norm, expon, lognorm
from numpy import log


#Load data
register_matplotlib_converters()
filename = 'lab01_data_profiling/data/set1_NYC_collisions_tabular.csv'
data = read_csv(filename, index_col='CRASH_DATE', na_values='', parse_dates=True, infer_datetime_format=True)

#Show details about range, min, max etc.
print(data.describe())

#Charts with details about numeric data in one chart
data.boxplot(rot=45)
savefig('lab01_data_profiling/images/data_distribution_set1_numeric_data_details.png')

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
savefig('lab01_data_profiling/images/data_distribution_set1_single_numeric_data_details.png')
show()


#Chart with numbers of outliers
NR_STDEV: int = 2

numeric_vars = ['PERSON_AGE']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

outliers_iqr = []
outliers_stdev = []
summary5 = data.describe(include='number')

for var in numeric_vars:
    iqr = 1.5 * (summary5[var]['75%'] - summary5[var]['25%'])
    outliers_iqr += [
        data[data[var] > summary5[var]['75%']  + iqr].count()[var] +
        data[data[var] < summary5[var]['25%']  - iqr].count()[var]]
    std = NR_STDEV * summary5[var]['std']
    outliers_stdev += [
        data[data[var] > summary5[var]['mean'] + std].count()[var] +
        data[data[var] < summary5[var]['mean'] - std].count()[var]]

outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
figure(figsize=(12, HEIGHT))
multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables', ylabel='nr outliers', percentage=False)
savefig('lab01_data_profiling/images/data_distribution_set1_outliers.png')
show()


#Histrograms with number of outliers
numeric_vars = ['PERSON_AGE']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Histogram for %s'%numeric_vars[n])
    axs[i, j].set_xlabel(numeric_vars[n])
    axs[i, j].set_ylabel("nr records")
    axs[i, j].hist(data[numeric_vars[n]].dropna().values, 'auto')
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('lab01_data_profiling/images/data_distribution_set1_single_histograms_numeric.png')
show()


#Histograms with trends
numeric_vars = ['PERSON_AGE']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    axs[i, j].set_title('Histogram with trend for %s'%numeric_vars[n])
    distplot(data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('lab01_data_profiling/images/data_distribution_set1_histograms_trend_numeric.png')
show()


#Histograms with distribution
def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = norm.pdf(x_values, mean, sigma)
    # Exponential
    loc, scale = expon.fit(x_values)
    distributions['Exp(%.2f)'%(1/scale)] = expon.pdf(x_values, loc, scale)
    # LogNorm
    sigma, loc, scale = lognorm.fit(x_values)
    distributions['LogNor(%.1f,%.2f)'%(log(scale),sigma)] = lognorm.pdf(x_values, sigma, loc, scale)
    return distributions

def histogram_with_distributions(ax: Axes, series: Series, var: str):
    values = series.sort_values().values
    ax.hist(values, 20, density=True)
    distributions = compute_known_distributions(values)
    multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')

numeric_vars = ['PERSON_AGE']
if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')

fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(numeric_vars)):
    histogram_with_distributions(axs[i, j], data[numeric_vars[n]].dropna(), numeric_vars[n])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('lab01_data_profiling/images/data_distribution_set1_histogram_numeric_distribution.png')
show()

#Historgrams for symbolic variables
symbolic_vars = ['CRASH_TIME', 'BODILY_INJURY', 'SAFETY_EQUIPMENT', 'PERSON_SEX', 'PERSON_TYPE', 'PED_LOCATION', 'CONTRIBUTING_FACTOR_2', 'EJECTION', 'COMPLAINT', 'EMOTIONAL_STATUS', 'CONTRIBUTING_FACTOR_1', 'POSITION_IN_VEHICLE', 'PED_ROLE', 'PED_ACTION'] 
if [] == symbolic_vars:
    raise ValueError('There are no symbolic variables.')

rows, cols = choose_grid(len(symbolic_vars))
fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
i, j = 0, 0
for n in range(len(symbolic_vars)):
    counts = data[symbolic_vars[n]].value_counts()
    bar_chart(counts.index.to_list(), counts.values, ax=axs[i, j], title='Histogram for %s'%symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig('lab01_data_profiling/images/data_distribution_set1_histograms_symbolic.png')
show()
