# first run knn on sample of unbalanced and unscaled database
# make sample

# change sample in percentage here
sample = 0.005

from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.neighbors import KNeighborsClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def KNN(file_tag, filename_train, filename_test, s):
    
    target = 'ALARM'

    train: DataFrame = read_csv(f'{filename_train}')
    train = train.sample(frac=s, replace=True, random_state=1)
    trnY: ndarray = train.pop(target).values
    trnX: ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    test: DataFrame = read_csv(f'{filename_test}')
    tstY: ndarray = test.pop(target).values
    tstX: ndarray = test.values

    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    values = {}
    best = (0, '')
    last_best = 0
    f1_scores_test = []
    f1_scores_train = []
    for d in dist:
        yvalues = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prdY = knn.predict(tstX)
            prdtrainY = knn.predict(trnX)
            #yvalues.append(accuracy_score(tstY, prdY))
            yvalues.append(f1_score(tstY, prdY))
            f1_scores_test.append(f1_score(tstY, prdY))
            f1_scores_train.append(f1_score(trnY, prdtrainY))
            if yvalues[-1] > last_best:
                best = (n, d)
                last_best = yvalues[-1]
        values[d] = yvalues

    x = nvalues
    y = f1_scores_test[:10]
    y2 = f1_scores_train[:10]

    fig, ax = plt.subplots()

    ax.plot(x, y)
    ax.plot(x, y2)
    ax.set_title('Overfit KNN')
    ax.set_xlabel('n')
    ax.set_ylabel('f1-score')
    #ax = set_elements(ax=ax, title= 'Overfit KNN', xlabel='n', ylabel='f1-score')
    ax.legend(['test', 'train'])
    savefig(f'lab05_decision_trees_and_overfitting\images\set2\{file_tag}_knn_overfit.png')
    plt.show()

    figure()
    multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel='accuracy', percentage=True)
    #savefig(f'lab03_knn_and_scaling\images\{file_tag}_knn_study.png')
    show()
    print('Best results with %d neighbors and %s'%(best[0], best[1]))

    clf = knn = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    #savefig(f'lab03_knn_and_scaling\images\{file_tag}_knn_best.png')
    show()

# Run KNN for all different codes NB: add test sets before running
#KNN('set2', 'lab04_naive_bayes_and_balancing\data\set2\set2_train.csv', 'lab04_naive_bayes_and_balancing\data\set2\set2_test.csv', sample)
#KNN('set2_scaled', 'lab04_naive_bayes_and_balancing\data\set2\set2_train_scaled.csv', 'lab04_naive_bayes_and_balancing\data\set2\set2_test_scaled.csv', sample)

test_set_scaled = 'lab04_naive_bayes_and_balancing\data\set2\set2_test_scaled.csv'
#KNN('set2_scaled_over', 'lab04_naive_bayes_and_balancing\data\set2\set2_over_scaled.csv', test_set_scaled, sample)
#KNN('set2_scaled_under', 'lab04_naive_bayes_and_balancing\data\set2\set2_under_scaled.csv', test_set_scaled, 1)
KNN('set2_scaled_smote', 'lab04_naive_bayes_and_balancing\data\set2\set2_smote_scaled.csv', test_set_scaled, sample)

#train: DataFrame = read_csv(f'{'lab04_naive_bayes_and_balancing\data\set1\scaled_over.csv'}')
#print(train.info())

test_set_unscaled = 'lab04_naive_bayes_and_balancing\data\set2\set2_test.csv'
# KNN('set2_unscaled_over', 'lab04_naive_bayes_and_balancing\data\set2\set2_over.csv', test_set_unscaled, sample)
# KNN('set2_unscaled_under', 'lab04_naive_bayes_and_balancing\data\set2\set2_under.csv', test_set_unscaled, 1)
# KNN('set2_unscaled_smote', 'lab04_naive_bayes_and_balancing\data\set2\set2_smote.csv', test_set_unscaled, sample)