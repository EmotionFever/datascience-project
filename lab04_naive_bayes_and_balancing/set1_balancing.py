# Commented out the class histogram, already have that one right?
# Dropped ID columns, why are they still here?

from pandas import read_csv
from matplotlib.pyplot import figure, savefig, show
#from ds_charts import bar_chart

#filename = 'data/unbalanced.csv'
filename = 'lab03_knn_and_scaling/ew_data/set1_scaled.csv'
file = "unbalanced"
original = read_csv(filename, sep=',', decimal='.')
class_var = 'PERSON_INJURY'
target_count = original[class_var].value_counts()
positive_class = target_count.idxmin()
negative_class = target_count.idxmax()
#ind_positive_class = target_count.index.get_loc(positive_class)
print('Minority class=', positive_class, ':', target_count[positive_class])
print('Majority class=', negative_class, ':', target_count[negative_class])
print('Proportion:', round(target_count[positive_class] / target_count[negative_class], 2), ': 1')
values = {'Original': [target_count[positive_class], target_count[negative_class]]}

# drop the ID's that are still in the dataset? why?
original = original.drop(columns=['VEHICLE_ID', 'PERSON_ID', 'UNIQUE_ID', 'COLLISION_ID', 'PED_ROLE'])
print(original.info())

#figure()
#bar_chart(target_count.index, target_count.values, title='Class balance')
#savefig(f'images/{file}_balance.png')
#show()

# split datasets per class 
df_positives = original[original[class_var] == positive_class]
df_negatives = original[original[class_var] == negative_class]


# undersample
from pandas import concat, DataFrame

df_neg_sample = DataFrame(df_negatives.sample(len(df_positives)))
df_under = concat([df_positives, df_neg_sample], axis=0)
df_under.to_csv(f'lab04_naive_bayes_and_balancing/data/{file}_under.csv', index=False)
values['UnderSample'] = [len(df_positives), len(df_neg_sample)]
print('Minority class=', positive_class, ':', len(df_positives))
print('Majority class=', negative_class, ':', len(df_neg_sample))
print('Proportion:', round(len(df_positives) / len(df_neg_sample), 2), ': 1')

# oversample
from pandas import concat, DataFrame

df_pos_sample = DataFrame(df_positives.sample(len(df_negatives), replace=True))
df_over = concat([df_pos_sample, df_negatives], axis=0)
df_over.to_csv(f'lab04_naive_bayes_and_balancing/data/{file}_over.csv', index=False)
values['OverSample'] = [len(df_pos_sample), len(df_negatives)]
print('Minority class=', positive_class, ':', len(df_pos_sample))
print('Majority class=', negative_class, ':', len(df_negatives))
print('Proportion:', round(len(df_pos_sample) / len(df_negatives), 2), ': 1')

# smote

from pandas import Series
from imblearn.over_sampling import SMOTE
RANDOM_STATE = 42

smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
y = original.pop(class_var).values
X = original.values
smote_X, smote_y = smote.fit_resample(X, y)
df_smote = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
df_smote.columns = list(original.columns) + [class_var]
df_smote.to_csv(f'lab04_naive_bayes_and_balancing/data/{file}_smote.csv', index=False)

smote_target_count = Series(smote_y).value_counts()
values['SMOTE'] = [smote_target_count[positive_class], smote_target_count[negative_class]]
print('Minority class=', positive_class, ':', smote_target_count[positive_class])
print('Majority class=', negative_class, ':', smote_target_count[negative_class])
print('Proportion:', round(smote_target_count[positive_class] / smote_target_count[negative_class], 2), ': 1')

from matplotlib.pyplot import figure, show
from ds_charts import multiple_bar_chart

figure()
multiple_bar_chart([positive_class, negative_class], values, title='Target', xlabel='frequency', ylabel='Class balance')
show()