from pandas import read_csv
from pandas import DataFrame

filename = 'lab02_data_preparation\ew_data\set1_mv_dummified.csv'
#filename = 'lab01_data_profiling\data\set1_NYC_collisions_tabular.csv'
data = read_csv(filename, na_values='?')
data.shape


THRESHOLD = 0.7

def select_redundant(corr_mtx, threshold: float): # -> tuple[dict, DataFrame]
    if corr_mtx.empty:
        return {}

    corr_mtx = abs(corr_mtx)
    vars_2drop = {}
    for el in corr_mtx.columns:
        el_corr = (corr_mtx[el]).loc[corr_mtx[el] >= threshold]
        if len(el_corr) == 1:
            corr_mtx.drop(labels=el, axis=1, inplace=True)
            corr_mtx.drop(labels=el, axis=0, inplace=True)
        else:
            vars_2drop[el] = el_corr.index
    return vars_2drop, corr_mtx

drop, corr_mtx = select_redundant(data.corr(), THRESHOLD)
#print(data.corr())
print(drop.keys())

from matplotlib.pyplot import figure, title, savefig, show
from seaborn import heatmap

if corr_mtx.empty:
    raise ValueError('Matrix is empty.')

figure(figsize=[10, 10])
heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=False, cmap='Blues')
title('Filtered Correlation Analysis')
savefig(f'lab06_random_forests_and_feature_selection\images\set1\set1_filtered_correlation_analysis_{THRESHOLD}.png')
show()


def drop_redundant(data: DataFrame, vars_2drop: dict) -> DataFrame:
    sel_2drop = []
    print(vars_2drop.keys())
    for key in vars_2drop.keys():
        if key not in sel_2drop:
            for r in vars_2drop[key]:
                if r != key and r not in sel_2drop:
                    sel_2drop.append(r)
    print('Variables to drop', sel_2drop)
    df = data.copy()
    for var in sel_2drop:
        df.drop(labels=var, axis=1, inplace=True)
    return df
df = drop_redundant(data, drop)
df

from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, get_variable_types

def select_low_variance(data: DataFrame, threshold: float) -> list:
    lst_variables = []
    lst_variances = []
    for el in data.columns:
        value = data[el].var()
        if value >= threshold:
            lst_variables.append(el)
            lst_variances.append(value)

    print(len(lst_variables), lst_variables)
    figure(figsize=[10, 4])
    bar_chart(lst_variables, lst_variances, title='Variance analysis', xlabel='variables', ylabel='variance')
    savefig('lab06_random_forests_and_feature_selection\images\set1\set_1filtered_variance_analysis.png')
    return lst_variables

numeric = get_variable_types(data)['Numeric']
vars_2drop = select_low_variance(data[numeric], 0.1)
print(vars_2drop)