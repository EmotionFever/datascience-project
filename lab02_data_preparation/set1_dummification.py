# Dummy encode the symbolic variables. The selection of symbolic variables is hardcoded 
# since selecting does not work. I also threw out some columns because they had to many missing values. 
# Retry this step after completing the missing values part. 

from pandas import read_csv
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
file = 'set1_NYC_collisions_tabular'
filename = 'lab01_data_profiling\data\set1_NYC_collisions_tabular.csv'
data = read_csv(filename, index_col='CRASH_DATE', na_values='', parse_dates=True, infer_datetime_format=True)

# Drop out all records with missing values
# This leaves no records at all. First drop columns with lot of empty cells. 
data = data.drop(['PED_LOCATION', 'CONTRIBUTING_FACTOR_2', 'CONTRIBUTING_FACTOR_1', 'PED_ACTION'], axis=1)
data.dropna(inplace=True)

from pandas import DataFrame, concat
#from ds_charts import get_variable_types
from sklearn.preprocessing import OneHotEncoder
from numpy import number

def dummify(df, vars_to_dummify):
    other_vars = [c for c in df.columns if not c in vars_to_dummify]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=bool)
    X = df[vars_to_dummify]
    encoder.fit(X)
    new_vars = encoder.get_feature_names(vars_to_dummify)
    trans_X = encoder.transform(X)
    dummy = DataFrame(trans_X, columns=new_vars, index=X.index)
    dummy = dummy.convert_dtypes(convert_boolean=True)

    final_df = concat([df[other_vars], dummy], axis=1)
    return final_df

#variables = get_variable_types(data)
#symbolic_vars = variables['Symbolic']

print(data.info())
#symbolic_vars = ['BODILY_INJURY', 'SAFETY_EQUIPMENT', 'PERSON_SEX', 'PERSON_TYPE', 'PED_LOCATION', 'CONTRIBUTING_FACTOR_2', 'EJECTION', 'COMPLAINT', 'EMOTIONAL_STATUS', 'CONTRIBUTING_FACTOR_1', 'POSITION_IN_VEHICLE', 'PED_ROLE', 'PED_ACTION', 'PERSON_INJURY']
symbolic_vars = ['BODILY_INJURY', 'SAFETY_EQUIPMENT', 'PERSON_SEX', 'PERSON_TYPE', 'EJECTION', 'COMPLAINT', 'EMOTIONAL_STATUS', 'POSITION_IN_VEHICLE', 'PED_ROLE', 'PERSON_INJURY']


print(symbolic_vars)
df = dummify(data, symbolic_vars)
df.to_csv(f'lab02_data_preparation\data\{file}_dummified.csv', index=False)

df.describe(include=[bool])