from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.neural_network import MLPClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT, plot_line
from sklearn.metrics import recall_score
from ds_charts import plot_overfitting_study

sample = 0.05

file_tag = 'set1'
filename = 'lab03_knn_and_scaling/ew_data/set1'
target = 'PERSON_INJURY'

train: DataFrame = read_csv(f'lab04_naive_bayes_and_balancing/ew_data/set1_scaled_smote.csv')
train = train.sample(frac=sample, replace=True, random_state=1)
trnY: ndarray = train.pop(target).values
trnX: ndarray = train.values
labels = unique(trnY)
labels.sort()

test: DataFrame = read_csv(f'{filename}_scaled_test.csv')
tstY: ndarray = test.pop(target).values
tstX: ndarray = test.values

lr_type = ['constant', 'invscaling', 'adaptive']
max_iter = [100, 300, 500, 750, 1000, 2500, 5000]
learning_rate = [.1, .5, .9]
best = ('', 0, 0)
last_best = 0
best_model = None
y_tst_values = []
y_trn_values = []
eval_metric = recall_score

cols = len(lr_type)
figure()
fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
for k in range(len(lr_type)):
    d = lr_type[k]
    values = {}
    for lr in learning_rate:
        yvalues = []
        y_tst_values_tmp = []
        y_trn_values_tmp = []
        for n in max_iter:
            mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=d,
                                learning_rate_init=lr, max_iter=n, verbose=False)
            mlp.fit(trnX, trnY)
            prdY = mlp.predict(tstX)
            prd_trn_Y = mlp.predict(trnX)
            yvalues.append(recall_score(tstY, prdY))
            y_tst_values_tmp.append(eval_metric(tstY, prdY))
            y_trn_values_tmp.append(eval_metric(trnY, prd_trn_Y))
            if yvalues[-1] > last_best:
                y_tst_values = y_tst_values_tmp
                y_trn_values = y_trn_values_tmp
                best = (d, lr, n)
                last_best = yvalues[-1]
                best_model = mlp
        values[lr] = yvalues
    multiple_line_chart(max_iter, values, ax=axs[0, k], title=f'MLP with lr_type={d}',
                           xlabel='mx iter', ylabel='recall', percentage=True)
savefig(f'lab07_neural_nets_and_gradient_boosting/images/{file_tag}/{file_tag}_mlp_study.png')
show()
print(f'Best results with lr_type={best[0]}, learning rate={best[1]} and {best[2]} max iter, with recall={last_best}')
########################################################

path_img = f'lab07_neural_nets_and_gradient_boosting/images/{file_tag}/{file_tag}_nn_overfit.png'
plot_overfitting_study(max_iter, y_trn_values, y_tst_values, name=f'Neural Network lr={best[1]}', xlabel='max_iterations', ylabel=str(eval_metric), path_img=path_img)
loss = best_model.loss_curve_
plot_line(xvalues=[i for i in range(len(loss))], yvalues=loss, xlabel='iterations', ylabel='loss', title='Loss function')
savefig(f'lab07_neural_nets_and_gradient_boosting/images/{file_tag}/{file_tag}_nn_loss.png')
show()
prd_trn = best_model.predict(trnX)
prd_tst = best_model.predict(tstX)
plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
savefig(f'lab07_neural_nets_and_gradient_boosting/images/{file_tag}/{file_tag}_nn_best.png')
show()