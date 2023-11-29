from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


bankData = pd.read_csv('../ModifiedDatasets/bank-additionalModified.csv', sep=';')

X_train = bankData.iloc[:,:-1].values
y_train = bankData.iloc[:, -1].values
X_test = bankData.iloc[:,:-1].values
y_test = bankData.iloc[:,-1].values

#Create a svm Classifier
clf = svm.SVC(kernel='rbf', gamma=0.1, C=10.0) # Linear Kernel doesnt work, needs to use rbf

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
