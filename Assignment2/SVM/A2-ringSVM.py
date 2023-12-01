from sklearn import svm
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import cross_val_score

ringTrainData = pd.read_csv('../A2-ring/A2-ring-separable.txt', sep='\t', header=None)
ringTestData = pd.read_csv('../A2-ring/A2-ring-test.txt', sep='\t', header=None)

X_train = ringTrainData.iloc[:,:-1].values
y_train = ringTrainData.iloc[:, -1].values
X_test = ringTestData.iloc[:,:-1].values
y_test = ringTestData.iloc[:,-1].values

# This is how the training set and testing set should look like when creating a model
#X_train = [[-0.137094, 0.899654], [0.542574,-0.492435], [-0.210166, 0.472680]]
#y_train = [0, 1,1]
#X_test = [[-0.2455, 0.411], [0.223,-0.1231], [-0.3211, 0.4221]]
#y_test = [1, 0,0]

#Create a svm Classifier
clf = svm.SVC(kernel='rbf', gamma=0.1, C=10.0) # Linear Kernel doesnt work, needs to use rbf

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
