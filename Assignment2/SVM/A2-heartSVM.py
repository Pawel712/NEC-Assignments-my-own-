from sklearn import svm
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


heartData = pd.read_csv('../ModifiedDatasets/HeartAttackModified.csv', sep=';')

X = heartData.iloc[:,:-1].values
y = heartData.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create a svm Classifier
clf = svm.SVC(kernel='rbf', gamma=0.1, C=10.0) 

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#print the error of the model
print("Classification Error on the test set:", 1 - metrics.accuracy_score(y_test, y_pred))

#-------------Cross-validation ----------------
num_folds = 5

crossValudationScore = cross_val_score(clf, X, y, cv=num_folds)

expectedClassificationError = 1 - crossValudationScore.mean()
print("Expected Classification Error:", expectedClassificationError)