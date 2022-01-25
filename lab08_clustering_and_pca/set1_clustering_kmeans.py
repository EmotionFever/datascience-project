from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import subplots, show, savefig
from ds_charts import choose_grid, plot_clusters, plot_line, plot_evaluation_results, compute_mae, compute_mse
from numpy import ndarray
from sklearn.metrics import davies_bouldin_score

sample = 0.05

file_tag = 'set1'
filename = 'lab03_knn_and_scaling/ew_data/set1'
target = 'PERSON_INJURY'

##### original set data

data: DataFrame = read_csv(f'lab03_knn_and_scaling/ew_data/{file_tag}_scaled.csv')
data = data.sample(frac=sample, replace=True, random_state=1)
# data.pop('id')
target_column = data.pop(target)
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
    labels = estimator.predict(data)
    if k == 2:
        best_model = estimator
    mse.append(compute_mse(data.values, labels, estimator.cluster_centers_))
    mae.append(compute_mae(data.values, labels, estimator.cluster_centers_))
    sc.append(silhouette_score(data, labels))
    dbs.append(davies_bouldin_score(data.values, labels))
    plot_clusters(data, v2, v1, estimator.labels_.astype(float), estimator.cluster_centers_, k, f'KMeans k={k}', ax=axs[i,j])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
savefig(f'lab08_clustering_and_pca/images/{file_tag}/{file_tag}_k-means_k_variation.png')
show()

fig, ax = subplots(1, 4, figsize=(10, 3), squeeze=False)
plot_line(N_CLUSTERS, mse, title='K-means MSE', xlabel='eps', ylabel='MSE', ax=ax[0, 0])
plot_line(N_CLUSTERS, mae, title='K-means MAE', xlabel='eps', ylabel='MAE', ax=ax[0, 1])
plot_line(N_CLUSTERS, sc, title='K-means SC', xlabel='eps', ylabel='SC', ax=ax[0, 2], percentage=True)
plot_line(N_CLUSTERS, dbs, title='K-means DBS', xlabel='eps', ylabel='DBS', ax=ax[0, 3])
savefig(f'lab08_clustering_and_pca/images/{file_tag}/{file_tag}_k-means_eval.png')
show()

train: DataFrame = read_csv(f'lab03_knn_and_scaling/ew_data/{file_tag}_scaled_train.csv')
train = train.sample(frac=sample, replace=True, random_state=1)
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv(f'lab03_knn_and_scaling/ew_data/{file_tag}_scaled_test.csv')
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values

prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
savefig(f'lab08_clustering_and_pca/images/{file_tag}/{file_tag}_k-means_best.png')
show()
mae = mean_absolute_error(target_column, best_model.labels_)
#mae = mae if mae > 0.5 else 1 - mae
print("MAE (k=2): " + str(mae))