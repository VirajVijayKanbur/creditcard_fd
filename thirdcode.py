import pandas as pd # Read the CSV file
df = pd.read_csv('creditcard.csv') # Show the contents
print(df)
#For getting more information than only the raw reords print(df.describe())
#include all the dependencies
from sklearn.model_selection import StratifiedShuffleSplit from sklearn.linear_model import LogisticRegression from sklearn.metrics import classification_report
import pandas as pd #read the data
data = pd.read_csv('creditcard.csv')
#select the features we would like to use during the training # Only use the 'Amount' and 'V1', ..., 'V28' features
features = ['A%d' % number for number in range(1, 14)]
# The target variable which we would like to predict, is the 'Class' variable target = 'A15'
# Now create an X variable (containing the features) and an y variable (containing only the target variable)
X = data[features] y = data[target] #normalization def normalize(X):
"""
 
Make the distribution of the values of each variable similar by subtracting the mean and by dividing by the standard deviation.
"""
for feature in X.columns: X[feature] -= X[feature].mean() X[feature] /= X[feature].std()
return X
# Define the model
model = LogisticRegression()
# Define the splitter for splitting the data in a train set and a test set
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0) # Loop through the splits (only one)
for train_indices, test_indices in splitter.split(X, y): // splitter generates indices
# Select the train and test data
X_train, y_train = X.iloc[train_indices], y.iloc[train_indices] X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
# Normalize the data
X_train = normalize(X_train) X_test = normalize(X_test)
# Fit and predict! model.fit(X_train, y_train) y_pred = model.predict(X_test) # And finally: show the results
print(classification_report(y_test, y_pred))
