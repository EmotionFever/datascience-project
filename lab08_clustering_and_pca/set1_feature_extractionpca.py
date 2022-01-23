from pandas import DataFrame, read_csv
from matplotlib.pyplot import figure, xlabel, ylabel, scatter, show, subplots, savefig
from sklearn.decomposition import PCA
from numpy.linalg import eig
from matplotlib.pyplot import gca, title

file_tag = 'set1'
filename = 'lab03_knn_and_scaling/ew_data/set1'
target = 'PERSON_INJURY'

data: DataFrame = read_csv(f'lab04_naive_bayes_and_balancing/ew_data/set1_smote_scaled.csv')
# data.pop('id')
data.pop(target)

variables = data.columns.values
eixo_x = 0
eixo_y = 1
eixo_z = 2

mean = (data.mean(axis=0)).tolist()
centered_data = data - mean
cov_mtx = centered_data.cov()
eigvals, eigvecs = eig(cov_mtx)

pca = PCA()
pca.fit(centered_data)
PC = pca.components_
var = pca.explained_variance_

# PLOT EXPLAINED VARIANCE RATIO
fig = figure(figsize=(4*3.5, 4))
title('Explained variance ratio')
xlabel('PC')
ylabel('ratio')
x_values = [str(i) for i in range(1, len(pca.components_) + 1)]
bwidth = 0.5
ax = gca()
ax.set_xticklabels(x_values)
ax.set_ylim(0.0, 1.0)
ax.bar(x_values, pca.explained_variance_ratio_, width=bwidth)
ax.plot(pca.explained_variance_ratio_)
for i, v in enumerate(pca.explained_variance_ratio_):
    ax.text(i, v+0.05, f'{v*100:.1f}', ha='center', fontweight='bold')
savefig(f'lab08_clustering_and_pca/images/{file_tag}/{file_tag}_pca_explained_variance_ratio.png')
show()

transf = pca.transform(data)

_, axs = subplots(1, 2, figsize=(2*5, 1*5), squeeze=False)
axs[0,0].set_xlabel(variables[eixo_y])
axs[0,0].set_ylabel(variables[eixo_z])
axs[0,0].scatter(data.iloc[:, eixo_y], data.iloc[:, eixo_z])

axs[0,1].set_xlabel('PC1')
axs[0,1].set_ylabel('PC2')
axs[0,1].scatter(transf[:, 0], transf[:, 1])
show()

column_values = ["PCA" + str(i) for i in range(1, transf.shape[1] + 1)]
print(column_values)
df_pca = DataFrame(data = transf, columns = column_values)
# Select the most important variables PCA1 to PCA6 for a 97.9 variance coverage
df_pca = df_pca[['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6']]
df_pca.to_csv(f'lab08_clustering_and_pca/ew_data/{file_tag}_pca.csv', index=False)