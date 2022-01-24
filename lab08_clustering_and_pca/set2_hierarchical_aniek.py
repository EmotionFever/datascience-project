from pandas import DataFrame, read_csv
from matplotlib.pyplot import subplots, show
from ds_charts import choose_grid, plot_clusters, plot_line, multiple_bar_chart, compute_mse, compute_centroids, compute_mae
from sklearn.metrics import davies_bouldin_score

sample = 0.25
#data: DataFrame = read_csv('lab08_clustering_and_pca\data\set1_pca.csv')
data: DataFrame = read_csv('lab03_knn_and_scaling\ew_data\set1_scaled.csv')
data = data.sample(frac=sample, replace=True, random_state=1)
#data.pop('id')
#data.pop('PERSON_INJURY')
data.reset_index(inplace=True, drop=True)
v1 = 0
v2 = 4

N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
rows, cols = choose_grid(len(N_CLUSTERS))

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

mse: list = []
mae: list = []
sc: list = []
dbs: list = []
rows, cols = choose_grid(len(N_CLUSTERS))
_, axs = subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
i, j = 0, 0
for n in range(len(N_CLUSTERS)):
    k = N_CLUSTERS[n]
    estimator = AgglomerativeClustering(n_clusters=k)
    estimator.fit(data)
    labels = estimator.labels_
    centers = compute_centroids(data, labels)
    mse.append(compute_mse(data.values, labels, centers))
    mae.append(compute_mae(data.values, labels, centers))
    sc.append(silhouette_score(data, labels))
    dbs.append(davies_bouldin_score(data.values, labels))
    plot_clusters(data, v2, v1, labels, centers, k, f'Hierarchical k={k}', ax=axs[i,j])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
show()

fig, ax = subplots(1, 4, figsize=(10, 3), squeeze=False)
plot_line(N_CLUSTERS, mse, title='DBSCAN MSE', xlabel='eps', ylabel='MSE', ax=ax[0, 0])
plot_line(N_CLUSTERS, mae, title='DBSCAN MAE', xlabel='eps', ylabel='MAE', ax=ax[0, 1])
plot_line(N_CLUSTERS, sc, title='DBSCAN SC', xlabel='eps', ylabel='SC', ax=ax[0, 2], percentage=True)
plot_line(N_CLUSTERS, sc, title='DBSCAN DBS', xlabel='eps', ylabel='DBS', ax=ax[0, 3])
show()

METRICS = ['euclidean', 'cityblock', 'chebyshev', 'cosine', 'jaccard']
LINKS = ['complete', 'average']
k = 3
values_mse = {}
values_mae = {}
values_sc = {}
values_dbs = {}
rows = len(METRICS)
cols = len(LINKS)
_, axs = subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
for i in range(len(METRICS)):
    mse: list = []
    mae: list = []
    sc: list = []
    dbs: list = []
    m = METRICS[i]
    for j in range(len(LINKS)):
        link = LINKS[j]
        estimator = AgglomerativeClustering(n_clusters=k, linkage=link, affinity=m )
        estimator.fit(data)
        labels = estimator.labels_
        centers = compute_centroids(data, labels)
        mse.append(compute_mse(data.values, labels, centers))
        mae.append(compute_mae(data.values, labels, centers))
        sc.append(silhouette_score(data, labels))
        dbs.append(davies_bouldin_score(data.values, labels))
        plot_clusters(data, v2, v1, labels, centers, k, f'Hierarchical k={k} metric={m} link={link}', ax=axs[i,j])
    values_mse[m] = mse
    values_mae[m] = mae
    values_sc[m] = sc
    values_dbs[m] = dbs
show()

_, ax = subplots(1, 4, figsize=(12, 3), squeeze=False)
multiple_bar_chart(LINKS, values_mse, title=f'Hierarchical MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
multiple_bar_chart(LINKS, values_mae, title=f'Hierarchical MAE', xlabel='metric', ylabel='MAE', ax=ax[0, 1])
multiple_bar_chart(LINKS, values_sc, title=f'Hierarchical SC', xlabel='metric', ylabel='SC', ax=ax[0, 2], percentage=True)
multiple_bar_chart(LINKS, values_dbs, title=f'Hierarchical DBS', xlabel='metric', ylabel='DBS', ax=ax[0, 3])
show()
