from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.ensemble import RandomForestClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

# change sample in percentage here
sample = 0.05

file_tag = 'set1'
filename = 'lab03_knn_and_scaling/ew_data/set1'
target = 'PERSON_INJURY'

train: DataFrame = read_csv(f'lab04_naive_bayes_and_balancing\data\set1\scaled_smote.csv')
train = train.sample(frac=sample, replace=True, random_state=1)
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv(f'{filename}_test_scaled.csv')
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values

n_estimators = [50]
max_depths = [5, 10, 25, 30]
max_features = [.3]
best = ('', 0, 0)
last_best = 0
best_model = None

cols = len(max_depths)
figure()
fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
recall_scores_test = []
recall_scores_train = []
for k in range(len(max_depths)):
    d = max_depths[k]
    values = {}
    for f in max_features:
        yvalues = []
        for n in n_estimators:
            rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
            rf.fit(trnX, trnY)
            prdY = rf.predict(tstX)
            prdtrainY = rf.predict(trnX)

            
            recall_scores_test.append(recall_score(tstY, prdY))
            recall_scores_train.append(recall_score(trnY, prdtrainY))

            yvalues.append(recall_score(tstY, prdY))
            if yvalues[-1] > last_best:
                best = (d, f, n)
                last_best = yvalues[-1]
                best_model = rf

        values[f] = yvalues
    
    

    

    multiple_line_chart(n_estimators, values, ax=axs[0, k], title=f'Random Forests with max_depth={d}',
                           xlabel='nr estimators', ylabel='recall', percentage=True)
savefig(f'lab06_random_forests_and_feature_selection/images/{file_tag}/{file_tag}_rf_study_recall.png')
show()
print('Best results with depth=%d, %1.2f features and %d estimators, with recall=%1.2f'%(best[0], best[1], best[2], last_best))

x = max_depths
y = recall_scores_test
y2 = recall_scores_train

    
fig, ax = plt.subplots()

ax.plot(x, y)
ax.plot(x, y2)
ax.set_title('Overfit random forest')
ax.set_xlabel('max dephts')
ax.set_ylabel('recall')
#ax = set_elements(ax=ax, title= 'Overfit KNN', xlabel='n', ylabel='f1-score')
ax.legend(['test', 'train'])
savefig(f'lab06_random_forests_and_feature_selection\images{file_tag}rf_overfit_depths.png')
plt.show()

prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
savefig(f'lab06_random_forests_and_feature_selection/images/{file_tag}/{file_tag}_rf_best.png')
show()

from numpy import std, argsort

variables = train.columns
importances = best_model.feature_importances_
stdevs = std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
indices = argsort(importances)[::-1]
elems = []
for f in range(len(variables)):
    elems += [variables[indices[f]]]
    print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

figure()
horizontal_bar_chart(elems, importances[indices], stdevs[indices], title='Random Forest Features importance', xlabel='importance', ylabel='variables')
savefig(f'lab06_random_forests_and_feature_selection/images/{file_tag}/{file_tag}_rf_ranking.png')
show()
