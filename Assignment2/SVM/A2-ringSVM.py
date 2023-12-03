from sklearn import svm
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import cross_val_score

ringTrainData = pd.read_csv('../A2-ring/A2-ring-separable.txt', sep='\t', header=None)
ringTestData = pd.read_csv('../A2-ring/A2-ring-test.txt', sep='\t', header=None)

X = ringTrainData.iloc[:,:-1].values
y = ringTrainData.iloc[:,-1].values

X_train = ringTrainData.iloc[:,:-1].values
y_train = ringTrainData.iloc[:, -1].values
X_test = ringTestData.iloc[:,:-1].values
y_test = ringTestData.iloc[:,-1].values

# This is how the training set and testing set should look like when creating a model
#X_train = [[-0.137094, 0.899654], [0.542574,-0.492435], [-0.210166, 0.472680]]
#y_train = [0, 1,1]
#X_test = [[-0.2455, 0.411], [0.223,-0.1231], [-0.3211, 0.4221]]
#y_test = [1, 0,0]

clf = svm.SVC(kernel='rbf', gamma=0.1, C=10.0) # Linear Kernel doesnt work, needs to use rbf

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# ------------------ classification formula ----------------

# Initialize counters for n_00, n_01, n_10, and n_11
n_00, n_01, n_10, n_11 = 0, 0, 0, 0

# Iterate through true labels and predictions to count occurrences
for true_label, predicted_label in zip(y_test, y_pred):
    if true_label == 0 and predicted_label == 0:
        n_00 += 1
    elif true_label == 0 and predicted_label == 1:
        n_01 += 1
    elif true_label == 1 and predicted_label == 0:
        n_10 += 1
    elif true_label == 1 and predicted_label == 1:
        n_11 += 1

# Compute the classification error percentage
classification_error = 100 * ((n_01 + n_10) / (n_00 + n_11 + n_01 + n_10))
print("Classification Error on the test set (%):", classification_error)
# ------------------ classification formula ----------------


#print error:
print("Classification Error on the test set:", 1 - metrics.accuracy_score(y_test, y_pred))

#-------------Cross-validation ----------------
num_folds = 5

crossValudationScore = cross_val_score(clf, X, y, cv=num_folds)

expectedClassificationError = 1 - crossValudationScore.mean()
print("Expected Classification Error:", expectedClassificationError)
#-------------Cross-validation ends ----------------
