import sys import numpy import pandas import matplotlib import seaborn import scipy
data = pd.read_csv('creditcard.csv')
data = data.sample(frac=0.1, random_state = 1) print(data.shape)
print(data.describe())
Fraud = data[data['Class'] == 1] Valid = data[data['Class'] == 0]
outlier_fraction = len(Fraud)/float(len(Valid)) print(outlier_fraction)
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1]))) 
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0]))) 
columns = data.columns.tolist()
columns = [c for c in columns if c not in ["Class"]] target = "Class"
X = data[columns] Y = data[target]
from sklearn.metrics import classification_report, accuracy_score from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
 
state = 1


classifiers = {
"Isolation Forest": IsolationForest(max_samples=len(X),
contamination=outlier_fraction, random_state=state),
"Local Outlier Factor": LocalOutlierFactor( n_neighbors=20, contamination=outlier_fraction)}
plt.figure(figsize=(9, 7)) n_outliers = len(Fraud)
for i, (clf_name, clf) in enumerate(classifiers.items()):


if clf_name == "Local Outlier Factor": y_pred = clf.fit_predict(X)
scores_pred = clf.negative_outlier_factor_ else:
clf.fit(X)
scores_pred = clf.decision_function(X) y_pred = clf.predict(X)
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1 n_errors = (y_pred != Y).sum()

print('{}: {}'.format(clf_name, n_errors)) print(accuracy_score(Y, y_pred))
print('Accuracy percentage:',accuracy_score(Y,y_pred)*100) print(classification_report(Y, y_pred))
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))


sns.heatmap(corrmat, vmax = .8, square = True) plt.show()
data.hist(figsize = (20, 20)) plt.show()
