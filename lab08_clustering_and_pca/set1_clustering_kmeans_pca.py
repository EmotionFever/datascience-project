from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import subplots, show, savefig
from ds_charts import choose_grid, plot_clusters, plot_line, plot_evaluation_results
from numpy import ndarray

sample = 0.05

file_tag = 'set1'
filename = 'lab03_knn_and_scaling/ew_data/set1'

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
sc: list = []
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
    mse.append(estimator.inertia_)
    sc.append(silhouette_score(data, estimator.labels_))
    plot_clusters(data, v2, v1, estimator.labels_.astype(float), estimator.cluster_centers_, k, f'KMeans k={k} PCA', ax=axs[i,j])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig(f'lab08_clustering_and_pca/images/{file_tag}/{file_tag}_k-means_k_variation_pca.png')
show()

fig, ax = subplots(1, 2, figsize=(6, 3), squeeze=False)
plot_line(N_CLUSTERS, mse, title='KMeans MSE PCA', xlabel='k', ylabel='MSE', ax=ax[0, 0])
plot_line(N_CLUSTERS, sc, title='KMeans SC PCA', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)
savefig(f'lab08_clustering_and_pca/images/{file_tag}/{file_tag}_k-means_eval_pca.png')
show()