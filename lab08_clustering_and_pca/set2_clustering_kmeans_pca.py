from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import subplots, show, savefig
from ds_charts import choose_grid, plot_clusters, plot_line, plot_evaluation_results, compute_mae
from numpy import ndarray
from sklearn.metrics import davies_bouldin_score

sample = 0.05

file_tag = 'set2'
filename = 'lab03_knn_and_scaling/ew_data/set2'

##### original set data
data: DataFrame = read_csv(f'lab08_clustering_and_pca/ew_data/{file_tag}_pca.csv')
data = data.sample(frac=sample, replace=True, random_state=1)

v1 = 0
v2 = 1

N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
rows, cols = choose_grid(len(N_CLUSTERS))

#### K-means

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error

mse: list = []
mae: list = []
sc: list = []
dbs: list = []
best_model = None
fig, axs = subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
i, j = 0, 0
iteration = 0
total_iteration = len(N_CLUSTERS)
for n in range(len(N_CLUSTERS)):
    k = N_CLUSTERS[n]
    iteration += 1
    print(">> " + str(iteration) + "/" + str(total_iteration) + " in k = " + str(k))
    estimator = KMeans(n_clusters=k)
    estimator.fit(data)
    if k == 2:
        best_model = estimator
    labels = estimator.predict(data)
    mse.append(estimator.inertia_)
    mae.append(compute_mae(data.values, labels, estimator.cluster_centers_))
    sc.append(silhouette_score(data, estimator.labels_))
    dbs.append(davies_bouldin_score(data.values, labels))
    plot_clusters(data, v2, v1, estimator.labels_.astype(float), estimator.cluster_centers_, k, f'KMeans k={k} PCA', ax=axs[i,j])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig(f'lab08_clustering_and_pca/images/{file_tag}/{file_tag}_k-means_k_variation_pca.png')
show()

fig, ax = subplots(1, 4, figsize=(10, 3), squeeze=False)
plot_line(N_CLUSTERS, mse, title='K-means PCA MSE', xlabel='eps', ylabel='MSE', ax=ax[0, 0])
plot_line(N_CLUSTERS, mae, title='K-means PCA MAE', xlabel='eps', ylabel='MAE', ax=ax[0, 1])
plot_line(N_CLUSTERS, sc, title='K-means PCA SC', xlabel='eps', ylabel='SC', ax=ax[0, 2], percentage=True)
plot_line(N_CLUSTERS, dbs, title='K-means PCA DBS', xlabel='eps', ylabel='DBS', ax=ax[0, 3])
savefig(f'lab08_clustering_and_pca/images/{file_tag}/{file_tag}_k-means_eval_pca.png')
show()