from pandas import read_csv, concat, DataFrame
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig
from ds_charts import bar_chart, get_variable_types
from sklearn.impute import SimpleImputer
from numpy import nan

register_matplotlib_converters()
file = 'set1'
#filename = 'ew_data/set1_outliers.csv'
filename = 'lab02_data_preparation\ew_data\set1_outliers.csv'
data = read_csv(filename, parse_dates=True, infer_datetime_format=True)

mv = {}
figure()
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable', xlabel='variables', ylabel='nr missing values', rotation=True)
#savefig(f'images/{file}_missing_values_number.png')
data.describe(include='all')

# defines the number of records to discard entire columns
threshold = data.shape[0] * 0.84

missings = [c for c in mv.keys() if mv[c]>threshold]
df_drop_columna = data.drop(columns=missings, inplace=False)
print('Dropped variables', missings)

mv = {}
figure()
for var in df_drop_columna.columns:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable', xlabel='variables', ylabel='nr missing values', rotation=True)
#savefig(f'images/{file}_missing_values_number_drop_columns.png')


tmp_rest, tmp_nr, tmp_sb = None, None, None
variables_all = df_drop_columna.columns
variables_fill_NA = ['VEHICLE_ID', 'POSITION_IN_VEHICLE', 'EJECTION', 'SAFETY_EQUIPMENT']
variables_fill_mean = ['PERSON_AGE']
variables_rest = [var for var in variables_all if var not in variables_fill_NA + variables_fill_mean]

tmp_rest = df_drop_columna[variables_rest]

imp = SimpleImputer(strategy='constant', fill_value='NA', missing_values=nan, copy=True)
tmp_sb = DataFrame(imp.fit_transform(df_drop_columna[variables_fill_NA]), columns=variables_fill_NA)

mean_age = round(df_drop_columna['PERSON_AGE'].mean(),0)
imp = SimpleImputer(strategy='constant', fill_value=mean_age, missing_values=nan, copy=True)
tmp_nr = DataFrame(imp.fit_transform(df_drop_columna[variables_fill_mean]), columns=variables_fill_mean)

df_filled_NA = concat([tmp_rest, tmp_nr, tmp_sb], axis=1)
df_filled_NA.to_csv(f'lab02_data_preparation\ew_data\{file}_mv.csv', index=False)
df_filled_NA.describe(include='all')
df_filled_NA