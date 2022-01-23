from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.tree import DecisionTreeClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, plot_line
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

file_tag = 'set1'
filename = 'lab03_knn_and_scaling/ew_data/set1'
target = 'PERSON_INJURY'

#train: DataFrame = read_csv(f'lab04_naive_bayes_and_balancing/data/scaled_smote.csv')
train: DataFrame = read_csv(f'lab04_naive_bayes_and_balancing/ew_data/set1_scaled_smote.csv')

trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv(f'{filename}_scaled_test.csv')
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values

min_impurity_decrease = [0.0001]
max_depths = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45]
criteria = ['entropy', 'gini']
best = ('',  0, 0.0)
last_best = 0
best_model = None

figure()
fig, axs = subplots(1, 2, figsize=(16, 4), squeeze=False)
f1_scores_test = []
f1_scores_train = []
for k in range(len(criteria)):
    f = criteria[k]
    values = {}
    for d in max_depths:
        yvalues = []
        for imp in min_impurity_decrease:
            tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
            tree.fit(trnX, trnY)
            prdY = tree.predict(tstX)
            prdtrainY = tree.predict(trnX)
            #yvalues.append(accuracy_score(tstY, prdY))
            yvalues.append(f1_score(tstY, prdY))
            f1_scores_test.append(f1_score(tstY, prdY))
            f1_scores_train.append(f1_score(trnY, prdtrainY))
            yvalues.append(f1_score(tstY, prdY))
            if yvalues[-1] > last_best:
                best = (f, d, imp)
                last_best = yvalues[-1]
                best_model = tree

        values[d] = yvalues
    #multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title=f'Decision Trees with {f} criteria',
    #                       xlabel='min_impurity_decrease', ylabel='precision_score', percentage=True)

    x = max_depths
    y = f1_scores_test[:10]
    y2 = f1_scores_train[:10]

    fig, ax = plt.subplots()

    ax.plot(x, y)
    ax.plot(x, y2)
    ax.set_title('Overfit decision tree')
    ax.set_xlabel('tree depth')
    ax.set_ylabel('f1-score')
    #ax = set_elements(ax=ax, title= 'Overfit KNN', xlabel='n', ylabel='f1-score')
    ax.legend(['test', 'train'])
    savefig(f'lab05_decision_trees_and_overfitting\images\set1\{file_tag}_tree_overfit.png')

    plt.show()
    
savefig(f'lab05_decision_trees_and_overfitting/images/{file_tag}/{file_tag}_dt_study_precision.png')
show()
print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.6f ==> accuracy=%1.2f'%(best[0], best[1], best[2], last_best))

# from sklearn import tree

# labels = [str(value) for value in labels]
# tree.plot_tree(best_model, feature_names=train.columns, class_names=labels)
# savefig(f'lab05_decision_trees_and_overfitting/images/{file_tag}/{file_tag}_dt_best_tree.png')
# show()

prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
savefig(f'lab05_decision_trees_and_overfitting/images/{file_tag}/{file_tag}_dt_best.png')
show()

from numpy import argsort, arange
from ds_charts import horizontal_bar_chart
from matplotlib.pyplot import Axes

variables = train.columns
importances = best_model.feature_importances_
indices = argsort(importances)[::-1]
elems = []
imp_values = []
for f in range(len(variables)):
    elems += [variables[indices[f]]]
    imp_values += [importances[indices[f]]]
    print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

figure()
horizontal_bar_chart(elems, imp_values, error=None, title='Decision Tree Features importance', xlabel='importance', ylabel='variables')
savefig(f'lab05_decision_trees_and_overfitting/images/{file_tag}/{file_tag}_dt_ranking.png')
show()
