from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, get_variable_types
from pandas import DataFrame

#Load data
register_matplotlib_converters()
file = 'set2'
filename = 'lab03_knn_and_scaling\ew_data\set2_ready_to_scale.csv'
data = read_csv(filename, na_values='', parse_dates=True, infer_datetime_format=True)
print(data.shape)


#Split data into each type
from ds_charts import get_variable_types

variable_types = get_variable_types(data)
numeric_vars = variable_types['Numeric']
symbolic_vars = variable_types['Symbolic']
boolean_vars = variable_types['Binary']

df_nr = data[numeric_vars]
df_sb = data[symbolic_vars]
df_bool = data[boolean_vars]

#Scaling z-score
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame, concat

transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
norm_data_zscore = concat([tmp, df_sb,  df_bool], axis=1)
# norm_data_zscore.to_csv(f'lab03_knn_and_scaling/ew_data/{file}_scaled_zscore.csv', index=False)

#Scaling MinMax normalization
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame, concat

transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
norm_data_minmax = concat([tmp, df_sb,  df_bool], axis=1)
# norm_data_minmax.to_csv(f'lab03_knn_and_scaling/ew_data/{file}_scaled_minmax.csv', index=False)
print(norm_data_minmax.describe())

#Boxplots
from matplotlib.pyplot import subplots, show

fig, axs = subplots(1, 3, figsize=(20,10),squeeze=False)
axs[0, 0].set_title('Original data')
data.boxplot(ax=axs[0, 0])
axs[0, 1].set_title('Z-score normalization')
norm_data_zscore.boxplot(ax=axs[0, 1])
axs[0, 2].set_title('MinMax normalization')
norm_data_minmax.boxplot(ax=axs[0, 2])
savefig('lab03_knn_and_scaling/images/set2/boxplots_with_scaling.png')
show()

fig, axs = subplots(1, 1, figsize=(20,10),squeeze=False)
axs[0, 0].set_title('Original data')
data.boxplot(ax=axs[0, 0])
savefig('lab03_knn_and_scaling/images/set2/oryginal_boxplots_with_scaling.png')
show()

fig, axs = subplots(1, 1, figsize=(20,10),squeeze=False)
axs[0, 0].set_title('Z-score normalization')
norm_data_zscore.boxplot(ax=axs[0, 0])
savefig('lab03_knn_and_scaling/images/set2/zscore_boxplots_with_scaling.png')
show()

fig, axs = subplots(1, 1, figsize=(20,10),squeeze=False)
axs[0, 0].set_title('MinMax normalization')
norm_data_minmax.boxplot(ax=axs[0, 0])
savefig('lab03_knn_and_scaling/images/set2/minmax_boxplots_with_scaling.png')
show()


norm_data_zscore.to_csv('lab03_knn_and_scaling/ew_data/set2_scaled.csv', index=False)
