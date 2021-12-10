import numpy as np
from pandas import read_csv, concat, unique, DataFrame
import matplotlib.pyplot as plt
import ds_charts as ds
from sklearn.model_selection import train_test_split

file_tag = 'set1'
data: DataFrame = read_csv('lab03_knn_and_scaling/data/set1_NYC_collisions_tabular.csv')
target = 'PERSON_INJURY'
positive = 0
negative = 1
positive_label = 'Injured'
negative_label = 'Killed'
values = {'Original': [len(data[data[target] == positive]), len(data[data[target] == negative])]}

y: np.ndarray = data.pop(target).values
X: np.ndarray = data.values
labels: np.ndarray = unique(y)
labels.sort()

# Balance data % https://towardsdatascience.com/how-to-balance-a-dataset-in-python-36dff9d12704
# Balacing technique used is Over Sample
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
over_sampler = RandomOverSampler(random_state=42)
X, y = over_sampler.fit_resample(X, y)
print(positive_label + ' is ' + str(positive))
print(negative_label + ' is ' + str(negative))
print(f"Training target statistics: {Counter(y)}")

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

train = concat([DataFrame(trnX, columns=data.columns), DataFrame(trnY,columns=[target])], axis=1)
train.to_csv(f'lab03_knn_and_scaling/ew_data/{file_tag}_train.csv', index=False)

test = concat([DataFrame(tstX, columns=data.columns), DataFrame(tstY,columns=[target])], axis=1)
test.to_csv(f'lab03_knn_and_scaling/ew_data/{file_tag}_test.csv', index=False)
values['Train'] = [len(np.delete(trnY, np.argwhere(trnY==negative))), len(np.delete(trnY, np.argwhere(trnY==positive)))]
values['Test'] = [len(np.delete(tstY, np.argwhere(tstY==negative))), len(np.delete(tstY, np.argwhere(tstY==positive)))]

plt.figure(figsize=(12,4))
ds.multiple_bar_chart([positive_label, negative_label], values, title='Data distribution per dataset')
plt.savefig(f'lab03_knn_and_scaling/images/{file_tag}_data_distribution_per_dataset.png')
plt.show()