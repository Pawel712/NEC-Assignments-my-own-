# This program creates linear regression model for the ring dataset

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict

ringTrainData = pd.read_csv('../A2-ring/A2-ring-separable.txt', sep='\t', header=None)
ringTestData = pd.read_csv('../A2-ring/A2-ring-test.txt', sep='\t', header=None)

X = ringTrainData.iloc[:,:-1].values
y = ringTrainData.iloc[:,-1].values

X_train = ringTrainData.iloc[:,:-1].values
y_train = ringTrainData.iloc[:, -1].values
X_test = ringTestData.iloc[:,:-1].values
y_test = ringTestData.iloc[:,-1].values

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

# ------------------- Classification error -------------

y_pred = model.predict(X_test)
# Convert predictions to binary values (0 or 1) based on a threshold (e.g., 0.5)
y_pred_binary = (y_pred > 0.5).astype(int)

n_00, n_01, n_10, n_11 = 0, 0, 0, 0

for true_label, predicted_label in zip(y_test, y_pred_binary):
    if true_label == 0 and predicted_label == 0:
        n_00 += 1
    elif true_label == 0 and predicted_label == 1:
        n_01 += 1
    elif true_label == 1 and predicted_label == 0:
        n_10 += 1
    elif true_label == 1 and predicted_label == 1:
        n_11 += 1

classification_error = 100 * ((n_01 + n_10) / (n_00 + n_11 + n_01 + n_10))
print("Classification Error (%):", classification_error)

# ------------------- Classification error ends -------------s

#------Cross validation ------
cv_predictions = cross_val_predict(model, X, y, cv=5)  # You can adjust the number of folds (cv) as needed
CrossValidationmae = mean_absolute_error(y_train, cv_predictions)

print("Cross validation error: ", CrossValidationmae)


