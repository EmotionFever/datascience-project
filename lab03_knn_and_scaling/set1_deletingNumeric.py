from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, get_variable_types
from pandas import DataFrame

#Load data
register_matplotlib_converters()
file = 'set1'
filename = 'lab02_data_preparation/ew_data/set1_mv_dummified.csv'
data = read_csv(filename, na_values='', parse_dates=True, infer_datetime_format=True)
print(data.shape)


#Split data into each type
from ds_charts import get_variable_types

variable_types = get_variable_types(data)
numeric_vars = variable_types['Numeric']
symbolic_vars = variable_types['Symbolic']
boolean_vars = variable_types['Binary']

print(symbolic_vars)

data = data.drop(labels=["PERSON_ID", "UNIQUE_ID", "COLLISION_ID"], axis=1)

PED_ROLE_encoded = { # how severe are the complaints
    'Driver': 1,
    'Passenger': 0
}
data["PED_ROLE"].replace(PED_ROLE_encoded, inplace=True)


data.to_csv('lab03_knn_and_scaling/ew_data/set1_deleteIDs.csv', index=False)

