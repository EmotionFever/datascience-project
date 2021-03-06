from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.naive_bayes import GaussianNB
from ds_charts import plot_evaluation_results, bar_chart

file_tag = 'set2'
filename = 'lab03_knn_and_scaling/ew_data/set2'
target = 'ALARM'

train: DataFrame = read_csv(f'lab04_naive_bayes_and_balancing/ew_data/set2_smote.csv')
# train: DataFrame = read_csv(f'{filename}_train.csv')
# train: DataFrame = read_csv(f'{filename}_train.csv')
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

# test: DataFrame = read_csv(f'{filename}_test.csv')
test: DataFrame = read_csv(f'{filename}_test.csv')
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values

clf = GaussianNB()
clf.fit(trnX, trnY)
prd_trn = clf.predict(trnX)
prd_tst = clf.predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
# savefig(f'lab04_naive_bayes_and_balancing/images/{file_tag}/{file_tag}_nb_best.png')
savefig(f'lab04_naive_bayes_and_balancing/images/{file_tag}/{file_tag}_nb_best_balancing_smote.png')
show()

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import accuracy_score

estimators = {'GaussianNB': GaussianNB(),
              'MultinomialNB': MultinomialNB(),
              'BernoulliNB': BernoulliNB()
              #'CategoricalNB': CategoricalNB
              }

xvalues = []
yvalues = []
for clf in estimators:
    xvalues.append(clf)
    estimators[clf].fit(trnX, trnY)
    prdY = estimators[clf].predict(tstX)
    yvalues.append(accuracy_score(tstY, prdY))

figure()
bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
savefig(f'lab04_naive_bayes_and_balancing/images/{file_tag}/{file_tag}_nb_study_balancing_smote.png')
# savefig(f'lab04_naive_bayes_and_balancing/images/{file_tag}/{file_tag}_nb_study.png')
show()