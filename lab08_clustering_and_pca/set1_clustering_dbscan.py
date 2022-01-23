from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import subplots, show, savefig
from ds_charts import choose_grid, plot_clusters, plot_line, plot_evaluation_results, compute_mse, compute_centroids, bar_chart
from numpy import ndarray

sample = 0.05

file_tag = 'set1'
filename = 'lab03_knn_and_scaling/ew_data/set1'
target = 'PERSON_INJURY'

##### original set data

data: DataFrame = read_csv(f'lab03_knn_and_scaling/ew_data/{file_tag}_scalled.csv')
data = data.sample(frac=sample, replace=True, random_state=1)
# data.pop('id')
target_column = data.pop(target)
v1 = 0
v2 = 1

N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
rows, cols = choose_grid(len(N_CLUSTERS))

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

EPS = [2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
mse: list = []
sc: list = []
rows, cols = choose_grid(len(EPS))
_, axs = subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
i, j = 0, 0
for n in range(len(EPS)):
    estimator = DBSCAN(eps=EPS[n], min_samples=2)
    estimator.fit(data)
    labels = estimator.labels_
    k = len(set(labels)) - (1 if -1 in labels else 0)
    if k > 1:
        centers = compute_centroids(data, labels)
        mse.append(compute_mse(data.values, labels, centers))
        sc.append(silhouette_score(data, labels))
        plot_clusters(data, v2, v1, labels.astype(float), estimator.components_, k, f'DBSCAN eps={EPS[n]} k={k}', ax=axs[i,j])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    else:
        mse.append(0)
        sc.append(0)
show()

fig, ax = subplots(1, 2, figsize=(6, 3), squeeze=False)
plot_line(EPS, mse, title='DBSCAN MSE', xlabel='eps', ylabel='MSE', ax=ax[0, 0])
plot_line(EPS, sc, title='DBSCAN SC', xlabel='eps', ylabel='SC', ax=ax[0, 1], percentage=True)
show()

import numpy as np
from scipy.spatial.distance import pdist, squareform

METRICS = ['euclidean', 'cityblock', 'chebyshev', 'cosine', 'jaccard']
distances = []
for m in METRICS:
    dist = np.mean(np.mean(squareform(pdist(data.values, metric=m))))
    distances.append(dist)

print('AVG distances among records', distances)
distances[0] *= 0.6
distances[1] = 80
distances[2] *= 0.6
distances[3] *= 0.1
distances[4] *= 0.15
print('CHOSEN EPS', distances)

mse: list = []
sc: list = []
rows, cols = choose_grid(len(METRICS))
_, axs = subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
i, j = 0, 0
for n in range(len(METRICS)):
    estimator = DBSCAN(eps=distances[n], min_samples=2, metric=METRICS[n])
    estimator.fit(data)
    labels = estimator.labels_
    k = len(set(labels)) - (1 if -1 in labels else 0)
    if k > 1:
        centers = compute_centroids(data, labels)
        mse.append(compute_mse(data.values, labels, centers))
        sc.append(silhouette_score(data, labels))
        plot_clusters(data, v2, v1, labels.astype(float), estimator.components_, k, f'DBSCAN metric={METRICS[n]} eps={distances[n]:.2f} k={k}', ax=axs[i,j])
    else:
        mse.append(0)
        sc.append(0)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
show()

fig, ax = subplots(1, 2, figsize=(6, 3), squeeze=False)
bar_chart(METRICS, mse, title='DBSCAN MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
bar_chart(METRICS, sc, title='DBSCAN SC', xlabel='metric', ylabel='SC', ax=ax[0, 1], percentage=True)
show()