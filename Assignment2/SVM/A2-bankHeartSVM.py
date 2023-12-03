from sklearn import svm
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#bank dataset
dataset = pd.read_csv('../ModifiedDatasets/bank-additionalModified.csv', sep=';')

#heartdataset#
#dataset = pd.read_csv('../ModifiedDatasets/HeartAttackModified.csv', sep=';')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = svm.SVC(kernel='rbf', gamma=0.1, C=10.0) 

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Classification Error on the test set:", 1 - metrics.accuracy_score(y_test, y_pred))

#-------------Cross-validation ----------------
num_folds = 5

crossValudationScore = cross_val_score(clf, X, y, cv=num_folds)

expectedClassificationError = 1 - crossValudationScore.mean()
print("Expected Classification Error:", expectedClassificationError)
#-------------Cross-validation ends ----------------


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
