from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, get_variable_types
from pandas import DataFrame

#Load data
register_matplotlib_converters()
filename = 'lab01_data_profiling/data/set1_NYC_collisions_tabular.csv'
data = read_csv(filename, index_col='CRASH_DATE', na_values='', parse_dates=True, infer_datetime_format=True)
print(data.shape)

#Chart with nr of records vs nr variables
figure(figsize=(4,2))
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
savefig('lab01_data_profiling/images/data_dimenstionality_set1_records_variables.png')
show()

#Data types
print(data.dtypes)

#Change object to category
cat_vars = data.select_dtypes(include='object')
data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
print(data.dtypes)

#Function to get variable types
def get_variable_types(df: DataFrame) -> dict:
    variable_types: dict = {
        'Numeric': [],
        'Binary': [],
        'Date': [],
        'Symbolic': []
    }
    for c in df.columns:
        uniques = df[c].dropna(inplace=False).unique()
        if len(uniques) == 2:
            variable_types['Binary'].append(c)
            df[c].astype('bool')
        elif df[c].dtype == 'datetime64':
            variable_types['Date'].append(c)
        elif df[c].dtype == 'int':
            variable_types['Numeric'].append(c)
        elif df[c].dtype == 'float':
            variable_types['Numeric'].append(c)
        else:
            df[c].astype('category')
            variable_types['Symbolic'].append(c)
    return variable_types


#Chart with nr of variables per type
variable_types = get_variable_types(data)
counts = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])
figure(figsize=(4,2))
bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
savefig('lab01_data_profiling/images/data_dimenstionality_set1_variable_types.png')
show()

#Chart with missing values
mv = {}
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

figure()
bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
            xlabel='variables', ylabel='nr missing values', rotation=True)
savefig('lab01_data_profiling/images/data_dimenstionality_set1_missing_values.png')
show()