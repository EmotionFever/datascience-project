from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from sklearn.naive_bayes import GaussianNB
from ds_charts import plot_evaluation_results, bar_chart

xvalues =  [ 'original', 'balanced', 'undersampled', 'oversampled', 'smote', 'balanced undersampled', 'balanced oversampled', 'balanced smote']
#yvalues =  [1,1, 0.96, 1, 1, 0.98, 1, 1]
yvalues =  [0.99 , 0.94, 0.94, 0.99, 0.98, 0.97, 0.95, 0.96]

figure()
bar_chart(xvalues, yvalues, title='Comparison of KNN Models', ylabel='accuracy', percentage=True)
savefig(f'lab04_naive_bayes_and_balancing\images')
show()